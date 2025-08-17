import os
import re
import sys
import time
import json
import math
import queue
import random
import signal
import socket
import zipfile
import inspect
import pathlib
import logging
import asyncio
import textwrap
import warnings
import datetime
import traceback
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

# --- Helper for dedup appending ---
def append_prediction_with_dedup(history, new_row, history_file=None, save_history_func=None):
    """
    Append new_row to history DataFrame while deduplicating by (ticker, interval, target_time).
    If an existing row with the same key exists it will be removed and replaced by new_row.
    If history is None or empty, returns a new DataFrame with new_row.
    If save_history_func is provided, it will be called with the resulting history for persistence.
    """
    import pandas as pd
    from zoneinfo import ZoneInfo

    if history is None or (hasattr(history, 'empty') and history.empty):
        out = pd.DataFrame([new_row])
    else:
        if 'ticker' not in history.columns:
            history['ticker'] = pd.NA
        if 'interval' not in history.columns:
            history['interval'] = pd.NA
        if 'target_time' not in history.columns:
            history['target_time'] = pd.NA

        # normalize target_time for comparison
        def _norm_ts(x):
            try:
                return str(ensure_timestamp_in_manila(x))
            except Exception:
                try:
                    import pandas as _pd
                except Exception:
                    _pd = __import__('pandas')
                return str(_pd.to_datetime(x, errors='coerce'))

        try:
            new_tgt_norm = _norm_ts(new_row.get('target_time'))
        except Exception:
            new_tgt_norm = str(new_row.get('target_time'))

        try:
            hist_ticker = history['ticker'].astype(str).str.upper()
            hist_interval = history['interval'].astype(str)
            hist_target_norm = history['target_time'].apply(lambda x: _norm_ts(x) if pd.notna(x) else "")
            new_key = (str(new_row.get('ticker','')).upper(), str(new_row.get('interval','')), str(new_tgt_norm))
            mask_same = (hist_ticker == new_key[0]) & (hist_interval == new_key[1]) & (hist_target_norm == new_key[2])
        except Exception:
            mask_same = pd.Series([False]*len(history), index=history.index)

        if mask_same.any():
            history = history.loc[~mask_same].reset_index(drop=True)

        missing_cols = set(new_row.keys()) - set(history.columns)
        for mc in missing_cols:
            history[mc] = pd.NA

        out = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True, sort=False)

    # try to save if requested
    try:
        if save_history_func is not None:
            save_history_func(out)
        elif history_file is not None:
            out.to_csv(history_file, index=False)
    except Exception:
        pass

    return out


# --- Patched evaluate_history ---
def evaluate_history(history_df, aligned_close_df, current_fetched_ts):
    """
    Evaluate unevaluated rows in history_df using aligned_close_df price data.
    - Only evaluates rows whose target_time is strictly in the past compared to current_fetched_ts.
    - Uses a small safety buffer (fraction of interval) so predictions made *very recently* are not evaluated.
    - Preference order for pred_price: stored pred_price -> price at predicted_at -> fetched_last_ts -> price just before target_time.
    - Neutral threshold is volatility-adaptive (rolling std of returns scaled).
    """
    import pandas as pd
    import numpy as np
    from zoneinfo import ZoneInfo

    if history_df is None or history_df.empty:
        return history_df

    # Ensure evaluated column exists
    if 'evaluated' not in history_df.columns:
        history_df['evaluated'] = False

    # normalize timestamp columns
    for c in ['predicted_at','fetched_last_ts','target_time','checked_at']:
        if c in history_df.columns:
            history_df[c] = pd.to_datetime(history_df[c], errors='coerce')

    # prepare aligned close and ensure Manila tz on index
    ac = aligned_close_df.copy()
    try:
        ac.index = pd.to_datetime(ac.index)
    except Exception:
        pass
    try:
        if getattr(ac.index, 'tz', None) is None:
            ac.index = ac.index.tz_localize('UTC', ambiguous='infer').tz_convert(ZoneInfo('Asia/Manila'))
        else:
            ac.index = ac.index.tz_convert(ZoneInfo('Asia/Manila'))
    except Exception:
        # fallback: leave naive index
        pass
    ac = ac.sort_index()
    ac.columns = [str(c).upper() for c in ac.columns]

    # determine now_ts (Manila)
    try:
        now_ts = ensure_timestamp_in_manila(current_fetched_ts)
    except Exception:
        now_ts = ensure_timestamp_in_manila(pd.Timestamp.now())

    # helper normalized columns
    history_df['_target_norm'] = history_df['target_time'].apply(lambda x: (ensure_timestamp_in_manila(x) if pd.notna(x) else pd.NaT))
    history_df['_predicted_at_norm'] = history_df['predicted_at'].apply(lambda x: (ensure_timestamp_in_manila(x) if pd.notna(x) else pd.NaT))

    # derive interval timedelta for safety buffer & vol window
    default_iv_td = pd.Timedelta(hours=1)
    iv_td = default_iv_td
    if 'interval' in history_df.columns and not history_df['interval'].isna().all():
        try:
            sample_iv = history_df['interval'].dropna().iloc[0]
            iv_td = interval_to_timedelta(sample_iv)
        except Exception:
            iv_td = default_iv_td

    # safety buffer: skip evaluation if predicted_at is within the last X% of the interval
    safety_buffer = pd.Timedelta(seconds=max(1, int(iv_td.total_seconds() * 0.05)))

    # mask: unevaluated, has target, and target_time < now_ts (strictly less)
    mask_ready = (~history_df['evaluated'].astype(bool)) & history_df['_target_norm'].notna() & (history_df['_target_norm'] < now_ts)

    # exclude rows where predicted_at is too recent
    recent_pred_mask = history_df['_predicted_at_norm'].notna() & (history_df['_predicted_at_norm'] > (now_ts - safety_buffer))
    mask_ready = mask_ready & (~recent_pred_mask)
    for idx in history_df[mask_ready].index:
        try:
            row = history_df.loc[idx]
            t_target = history_df.at[idx, '_target_norm']
            tk_raw = row.get('ticker', '')
            tk = str(tk_raw).upper().strip() if pd.notna(tk_raw) else ''
            history_df.at[idx, 'checked_at'] = pd.Timestamp.now(tz=ZoneInfo('Asia/Manila'))

            if not tk:
                history_df.at[idx, 'eval_error'] = 'missing_ticker'
                continue
            if tk not in ac.columns:
                history_df.at[idx, 'eval_error'] = f'ticker_not_in_price_matrix:{tk}'
                continue

            series = ac[tk].dropna()
            if series.empty:
                history_df.at[idx, 'eval_error'] = 'no_price_data_for_ticker'
                continue

            raw_label = row.get('predicted_label', None)
            pred_label = canonical_label(raw_label)
            if pred_label is None:
                history_df.at[idx, 'eval_error'] = f'ambiguous_predicted_label:{raw_label}'
                continue

            # determine prediction price (preference order)
            pred_price_val = None
            if pd.notna(row.get('pred_price')):
                try:
                    pred_price_val = float(row.get('pred_price'))
                except Exception:
                    pred_price_val = None

            if pred_price_val is None:
                pred_time_candidate = None
                if pd.notna(history_df.at[idx, '_predicted_at_norm']):
                    pred_time_candidate = history_df.at[idx, '_predicted_at_norm']
                elif pd.notna(row.get('fetched_last_ts')):
                    pred_time_candidate = ensure_timestamp_in_manila(row.get('fetched_last_ts'))
                else:
                    pred_time_candidate = t_target - pd.Timedelta(seconds=1)
                pred_price_val = price_at_or_before(series, pred_time_candidate)

            actual_price_val = price_at_or_before(series, t_target)

            history_df.at[idx, 'pred_price'] = (None if pd.isna(pred_price_val) else float(pred_price_val))
            history_df.at[idx, 'actual_price'] = (None if pd.isna(actual_price_val) else float(actual_price_val))

            if pd.isna(pred_price_val) or pd.isna(actual_price_val):
                history_df.at[idx, 'eval_error'] = 'missing_pred_or_actual_price_for_eval'
                history_df.at[idx, 'pct_change'] = None
                history_df.at[idx, 'correct'] = None
                continue

            if pred_price_val == 0:
                history_df.at[idx, 'eval_error'] = 'pred_price_zero'
                history_df.at[idx, 'pct_change'] = None
                history_df.at[idx, 'correct'] = None
                continue

            pct = (float(actual_price_val) - float(pred_price_val)) / float(pred_price_val)
            history_df.at[idx, 'pct_change'] = pct

            # compute rolling vol window based on interval (aim for ~24 hours worth of bars)
            try:
                iv_seconds = iv_td.total_seconds()
                bars_in_24h = max(1, int(24 * 3600 / max(1, iv_seconds)))
            except Exception:
                bars_in_24h = 24

            rolling_vol = series.pct_change().rolling(window=bars_in_24h, min_periods=1).std()
            # volatility reference time (prefer fetched_last_ts then predicted_at)
            vol_ref_time = None
            if pd.notna(row.get('fetched_last_ts')):
                vol_ref_time = ensure_timestamp_in_manila(row.get('fetched_last_ts'))
            elif pd.notna(history_df.at[idx, '_predicted_at_norm']):
                vol_ref_time = history_df.at[idx, '_predicted_at_norm']
            else:
                vol_ref_time = t_target - pd.Timedelta(seconds=1)

            vol_at_prediction = rolling_vol.asof(vol_ref_time)

            if pd.isna(vol_at_prediction) or vol_at_prediction == 0:
                thr = 0.002
            else:
                thr = float(vol_at_prediction) * 0.5

            history_df.at[idx, 'neutral_threshold_used'] = float(thr)

            if pct > thr:
                actual_movement_category = 'up'
            elif pct < -thr:
                actual_movement_category = 'down'
            else:
                actual_movement_category = 'neutral'

            history_df.at[idx, 'correct'] = (pred_label == actual_movement_category)
            history_df.at[idx, 'evaluated'] = True
            history_df.at[idx, 'eval_error'] = None

        except Exception as e:
            try:
                history_df.at[idx, 'eval_error'] = f'exception_during_eval:{str(e)}'
                history_df.at[idx, 'checked_at'] = pd.Timestamp.now(tz=ZoneInfo('Asia/Manila'))
            except Exception:
                pass
            continue

    # cleanup helper cols
    if '_target_norm' in history_df.columns:
        history_df = history_df.drop(columns=['_target_norm'])
    if '_predicted_at_norm' in history_df.columns:
        history_df = history_df.drop(columns=['_predicted_at_norm'])

    return history_df
def recompute_history_evaluations(history_df, aligned_close_df, save_back=False, path_if_saving=None):
    """
    Re-evaluate a history DataFrame using aligned_close_df. Repairs rows that were
    previously evaluated too early or computed incorrectly.
    - history_df: pandas DataFrame
    - aligned_close_df: DataFrame of close prices indexed by timestamp, cols=tickers
    - save_back: if True and path_if_saving provided, will save to CSV
    Returns: repaired DataFrame
    """
    import pandas as pd
    import numpy as np
    from zoneinfo import ZoneInfo

    if history_df is None or history_df.empty:
        return history_df

    MANILA = ZoneInfo("Asia/Manila")

    def parse_ts_to_manila(x):
        if pd.isna(x) or x == "":
            return pd.NaT
        ts = pd.to_datetime(x, errors="coerce")
        if pd.isna(ts):
            return pd.NaT
        if ts.tzinfo is None:
            try:
                return ts.tz_localize("UTC").tz_convert(MANILA)
            except Exception:
                return ts.tz_localize(MANILA)
        return ts.tz_convert(MANILA)

    # normalize timestamps in history
    for c in ['predicted_at','fetched_last_ts','target_time','checked_at']:
        if c in history_df.columns:
            history_df[c] = history_df[c].apply(parse_ts_to_manila)

    # ensure close price df tz-aware and sorted
    ac = aligned_close_df.copy()
    try:
        ac.index = pd.to_datetime(ac.index)
    except Exception:
        pass
    try:
        if getattr(ac.index, 'tz', None) is None:
            ac.index = ac.index.tz_localize("UTC").tz_convert(MANILA)
        else:
            ac.index = ac.index.tz_convert(MANILA)
    except Exception:
        try:
            ac.index = ac.index.tz_localize(MANILA)
        except Exception:
            pass
    ac = ac.sort_index()
    ac.columns = [str(c).upper() for c in ac.columns]

    now = pd.Timestamp.now(tz=MANILA)
    max_ts = ac.index.max() if not ac.empty else pd.NaT

    out_rows = []
    for i, row in history_df.iterrows():
        r = row.to_dict()
        tk = str(r.get('ticker','')).upper().strip()
        tgt = r.get('target_time')
        pred_at = r.get('predicted_at')

        if pd.isna(tk) or tk == '' or tk not in ac.columns:
            r['evaluated'] = False
            r['eval_error'] = 'ticker_not_in_price_matrix'
            out_rows.append(r)
            continue

        series = ac[tk].dropna()
        if series.empty:
            r['evaluated'] = False
            r['eval_error'] = 'no_price_data'
            out_rows.append(r)
            continue

        # recompute pred and actual prices
        pred_price = None
        if pd.notna(pred_at):
            pred_price = price_at_or_before(series, pred_at)
        else:
            # fallback to fetched_last_ts or just before target
            fetched_ts = r.get('fetched_last_ts')
            if pd.notna(fetched_ts):
                pred_price = price_at_or_before(series, fetched_ts)
            else:
                if pd.notna(tgt):
                    pred_price = price_at_or_before(series, tgt - pd.Timedelta(seconds=1))

        r['pred_price'] = (None if pd.isna(pred_price) else float(pred_price))

        actual_price = None
        if pd.notna(tgt) and pd.notna(max_ts) and max_ts >= tgt:
            actual_price = price_at_or_before(series, tgt)
        r['actual_price'] = (None if pd.isna(actual_price) else float(actual_price))

        # only mark evaluated if actual_price exists and target_time is in the past
        if pd.notna(actual_price) and pd.notna(pred_price) and now >= tgt:
            pct = (actual_price - pred_price) / pred_price if pred_price else np.nan
            if abs(pct) < 1e-12:
                pct = 0.0
            r['pct_change'] = pct

            # compute volatility-based threshold (24h window approximate)
            try:
                idx = series.index
                if len(idx) >= 2:
                    freq_seconds = max(1, int((idx[1] - idx[0]).total_seconds()))
                    bars = max(1, int(24*3600 / freq_seconds))
                else:
                    bars = 24
            except Exception:
                bars = 24

            rolling_vol = series.pct_change().rolling(window=bars, min_periods=1).std()
            vol_at_pred = rolling_vol.asof(pred_at) if pd.notna(pred_at) else rolling_vol.asof(tgt - pd.Timedelta(seconds=1))
            if pd.isna(vol_at_pred) or vol_at_pred == 0:
                thr = 0.002
            else:
                thr = float(vol_at_pred) * 0.5
            r['neutral_threshold_used'] = float(thr)

            lab = str(r.get('predicted_label','')).lower()
            if lab.startswith('u'):
                pred_lab = 'up'
            elif lab.startswith('d'):
                pred_lab = 'down'
            else:
                pred_lab = 'neutral'

            if pct > thr:
                actual_lab = 'up'
            elif pct < -thr:
                actual_lab = 'down'
            else:
                actual_lab = 'neutral'

            r['correct'] = (pred_lab == actual_lab)
            r['evaluated'] = True
            r['eval_error'] = None
        else:
            r['evaluated'] = False
            r['correct'] = None
            r['pct_change'] = np.nan
            r['eval_error'] = 'not_ready_for_eval'

        r['checked_at'] = pd.Timestamp.now(tz=MANILA)
        out_rows.append(r)

    out_df = pd.DataFrame(out_rows)

    # if requested, save
    if save_back:
        try:
            if path_if_saving:
                out_df.to_csv(path_if_saving, index=False)
            else:
                out_df.to_csv(history_file, index=False)
        except Exception:
            pass

    return out_df
# ======================================================
# Streamlit App: Stock Prediction & History Management
# ======================================================

st.set_page_config(
    page_title="Solaris Stock Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Utility Functions (some already existed in your code) ---

def ensure_timestamp_in_manila(x):
    """Convert timestamp to Asia/Manila tz-aware Timestamp."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return pd.NaT
    ts = pd.to_datetime(x, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    if ts.tzinfo is None:
        return ts.tz_localize("UTC").tz_convert(ZoneInfo("Asia/Manila"))
    return ts.tz_convert(ZoneInfo("Asia/Manila"))

def canonical_label(label):
    """Normalize prediction labels (up, down, neutral)."""
    if label is None or str(label).strip() == "":
        return None
    s = str(label).lower().strip()
    if s.startswith("u"):
        return "up"
    if s.startswith("d"):
        return "down"
    if s.startswith("n"):
        return "neutral"
    return None

def price_at_or_before(series, ts):
    """Return price from series at or before ts (asof lookup)."""
    if series is None or series.empty or pd.isna(ts):
        return None
    try:
        return series.asof(ts)
    except Exception:
        return None

def interval_to_timedelta(interval: str) -> pd.Timedelta:
    """Convert interval string to a timedelta (e.g. '1h', '30m')."""
    if not interval:
        return pd.Timedelta(hours=1)
    s = str(interval).lower().strip()
    if s.endswith("m"):
        return pd.Timedelta(minutes=int(s[:-1]))
    if s.endswith("h"):
        return pd.Timedelta(hours=int(s[:-1]))
    if s.endswith("d"):
        return pd.Timedelta(days=int(s[:-1]))
    return pd.Timedelta(hours=1)

# --- File paths ---
history_file = "prediction_history.csv"

# --- Load/save history ---
def load_history():
    try:
        df = pd.read_csv(history_file)
        return df
    except Exception:
        return pd.DataFrame()

def save_history(df):
    try:
        df.to_csv(history_file, index=False)
    except Exception as e:
        st.warning(f"Could not save history: {e}")

# --- Initialize history ---
history = load_history()

# --- Sidebar ---
st.sidebar.title("Controls")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"], index=4)
prediction_label = st.sidebar.selectbox("Predicted Direction", ["Up", "Down", "Neutral"])
target_offset = st.sidebar.number_input("Target offset (intervals ahead)", value=1, min_value=1, max_value=100)

if st.sidebar.button("Make Prediction"):
    target_time = pd.Timestamp.now(tz=ZoneInfo("Asia/Manila")) + interval_to_timedelta(interval) * target_offset
    new_row = {
        "ticker": ticker,
        "interval": interval,
        "predicted_label": prediction_label.lower(),
        "predicted_at": pd.Timestamp.now(tz=ZoneInfo("Asia/Manila")),
        "target_time": target_time,
        "fetched_last_ts": pd.Timestamp.now(tz=ZoneInfo("Asia/Manila")),
        "evaluated": False,
        "pred_price": None,
        "actual_price": None,
    }
    # use dedup-aware append
    history = append_prediction_with_dedup(history, new_row, history_file=history_file, save_history_func=save_history)
    st.success("Prediction saved!")
# ============================
# Recent Predictions Section
# ============================

st.subheader("📈 Recent Predictions")

# Auto-repair before rendering
try:
    if history is not None and not history.empty:
        # if we already have aligned_close from earlier logic, reuse it
        ac_for_repair = None
        if 'aligned_close' in globals():
            ac_for_repair = globals()['aligned_close']
        if ac_for_repair is not None:
            history = recompute_history_evaluations(history, ac_for_repair, save_back=False)
except Exception as e:
    st.warning(f"Auto-repair attempt failed: {e}")

# Manual repair button
if st.button("🔧 Repair History Now"):
    try:
        ac_for_repair = None
        if 'aligned_close' in globals():
            ac_for_repair = globals()['aligned_close']
        if ac_for_repair is None:
            st.warning("Repair requires price series (aligned_close) — run a prediction first or re-fetch data.")
        else:
            repaired = recompute_history_evaluations(history, ac_for_repair, save_back=True, path_if_saving=history_file)
            save_history(repaired)
            st.success("History repaired and saved.")
            history = repaired
    except Exception as e:
        st.error(f"Repair failed: {e}")

# Show recent predictions table
if history is not None and not history.empty:
    try:
        df_show = history.copy()
        # format timestamps
        for c in ['predicted_at','fetched_last_ts','target_time','checked_at']:
            if c in df_show.columns:
                df_show[c] = pd.to_datetime(df_show[c], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(df_show.tail(20), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to render history table: {e}")
else:
    st.info("No predictions yet. Add one from the sidebar.")
# ============================
# Evaluation Trigger
# ============================

st.subheader("🔍 Evaluate Predictions")

if st.button("Evaluate Now"):
    try:
        if history is not None and not history.empty:
            # If aligned_close is available in scope, pass it for evaluation
            ac_for_eval = None
            if 'aligned_close' in globals():
                ac_for_eval = globals()['aligned_close']
            if ac_for_eval is not None:
                history = evaluate_history(history, ac_for_eval, current_fetched_ts=pd.Timestamp.now(tz=ZoneInfo("Asia/Manila")))
                save_history(history)
                st.success("Evaluation complete!")
            else:
                st.warning("No aligned_close price data available. Fetch data first before evaluating.")
        else:
            st.info("No predictions to evaluate.")
    except Exception as e:
        st.error(f"Evaluation failed: {e}")

# ============================
# Footer
# ============================

st.markdown("---")
st.caption("Solaris Stock Predictor — Hybrid ML model for short-term trend forecasting")
