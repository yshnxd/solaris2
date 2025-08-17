# app.py - SOLARIS — Stock Price Movement Predictor (Revised: safer loading, cached downloads, robust history eval, VOLATILITY-ADAPTIVE NEUTRAL THRESHOLD, correct deferred evaluation)
import os
import tempfile
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import yfinance as yf
import ta

from datetime import datetime, timedelta
try:
    from zoneinfo import ZoneInfo     # Python 3.9+
except Exception:
    from pytz import timezone as ZoneInfo

import random, tensorflow as tf
from pathlib import Path

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

st.set_page_config(page_title="Solaris - Predict Next Hour!", layout="wide")
st.title("SOLARIS — Stock Price Movement Predictor")
st.markdown("_A machine learning–driven hybrid model for short-term market trend forecasting_")

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker (any Yahoo Finance symbol)", "AAPL")
ticker = ticker.strip().upper()

sequence_length = st.sidebar.number_input("Sequence length (LSTM/CNN)", min_value=3, max_value=240, value=24, step=1)
interval = st.sidebar.selectbox("Interval", ["60m"], index=0)
period_default = "90d" if interval == "60m" else "90d"
period = st.sidebar.text_input("Period (try 90d, 365d, or 729d)", period_default)

st.sidebar.markdown("---")
st.sidebar.write("Model files (relative to app.py):")
cnn_path   = st.sidebar.text_input("CNN (.pkl)", "cnn_model.pkl")
lstm_path  = st.sidebar.text_input("LSTM (.pkl)", "lstm_model.pkl")
xgb_path   = st.sidebar.text_input("XGB (.pkl)", "xgb_model.pkl")
meta_path  = st.sidebar.text_input("Meta (.pkl)", "meta_learner.pkl")

st.sidebar.markdown("---")
label_map_input = st.sidebar.text_input("Labels (index order)", "down,neutral,up")
label_map = [s.strip().lower() for s in label_map_input.split(",")]
prob_threshold = st.sidebar.slider("Prob threshold for confident class", 0.05, 0.95, 0.5, 0.05)
models_to_run = st.sidebar.multiselect("Models to run", ["cnn", "lstm", "xgb", "meta"], default=["cnn", "lstm", "xgb", "meta"])
run_button = st.sidebar.button("Fetch live data & predict NEXT interval (real-time)")
history_file = st.sidebar.text_input("History CSV file", "predictions_history.csv")

COLOR_BUY_BG = "#d7e9da"   # soft green
COLOR_SELL_BG = "#f0e5e6"  # soft rose
COLOR_HOLD_BG = "#f3f4f6"  # soft gray
COLOR_TEXT = "#111111"

def warn(msg):
    try:
        st.warning(msg)
    except Exception:
        print("WARNING:", msg)

@st.cache_resource(show_spinner=False)
def load_model_safe(path: str):
    if not path or not os.path.exists(path):
        warn(f"Model file not found: {path}")
        return None
    try:
        mod = joblib.load(path)
        return mod
    except Exception as e_joblib:
        try:
            from tensorflow.keras.models import load_model as keras_load_model
            try:
                mod = keras_load_model(path)
                return mod
            except Exception as e_keras:
                warn(f"keras loader failed for {path}: {e_keras} (joblib error: {e_joblib})")
                return None
        except Exception:
            warn(f"Model load failed for {path}: {e_joblib}")
            return None

@st.cache_data(ttl=30, show_spinner=False)
def download_price_cached(ticker: str, interval: str, period: str = None, start=None, end=None, use_tk_history=False):
    try:
        if use_tk_history:
            tk = yf.Ticker(ticker)
            df = tk.history(period=period, interval=interval, start=start, end=end, auto_adjust=False, prepost=False, actions=False)
        else:
            if start is not None or end is not None:
                df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
            else:
                df = yf.download(ticker, interval=interval, period=period, progress=False)
        if df is None:
            return pd.DataFrame()
        df = df.dropna(how='all')
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
        return df
    except Exception as e:
        warn(f"yf download failed for {ticker} ({interval}): {e}")
        return pd.DataFrame()

def save_history_atomic(df: pd.DataFrame, target_path: str):
    try:
        p = Path(target_path)
        if p.parent:
            p.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=str(p.parent), suffix=".csv") as tmp:
            df.to_csv(tmp.name, index=False)
            tmp_name = tmp.name
        os.replace(tmp_name, str(p))
    except Exception as e:
        warn(f"Failed to save history atomically to {target_path}: {e}")
        try:
            df.to_csv(target_path, index=False)
        except Exception as e2:
            warn(f"Fallback save also failed: {e2}")

def get_feature_names(mod):
    f = getattr(mod, "feature_names_in_", None)
    if f is not None:
        return list(f)
    alt = getattr(mod, "feature_columns", None)
    if alt is not None:
        return list(alt)
    return None

def get_n_features(mod):
    n = getattr(mod, "n_features_in_", None)
    if n is not None:
        return int(n)
    inp = getattr(mod, "input_shape", None)
    if inp is not None:
        try:
            if isinstance(inp, tuple):
                return int(inp[-1])
            if isinstance(inp, (list, tuple)):
                return int(inp[-1])
        except Exception:
            pass
    return None

def align_tabular_row_to_names(last_row_df, expected_names):
    out = pd.DataFrame(index=[0])
    for nm in expected_names:
        out[nm] = last_row_df.iloc[0][nm] if nm in last_row_df.columns else 0.0
    return out

def align_seq_df_to_names(seq_df, expected_names):
    X = seq_df.copy()
    for nm in expected_names:
        if nm not in X.columns:
            X[nm] = 0.0
    extra = [c for c in X.columns if c not in expected_names]
    if extra:
        X = X.drop(columns=extra)
    return X[expected_names]

def ensure_timestamp_in_manila(ts):
    try:
        tz = ZoneInfo("Asia/Manila")
    except Exception:
        from pytz import timezone
        tz = timezone("Asia/Manila")

    ts = pd.to_datetime(ts, errors='coerce')
    if pd.isna(ts):
        return pd.Timestamp.now(tz=tz)
    if getattr(ts, "tzinfo", None) is None:
        try:
            ts = ts.tz_localize("UTC", ambiguous='infer').tz_convert(tz)
        except Exception:
            try:
                ts = ts.tz_localize(tz, ambiguous='infer')
            except Exception:
                pass
    else:
        try:
            ts = ts.tz_convert(tz)
        except Exception:
            pass
    return ts

def compute_real_next_time_now(interval):
    try:
        tz = ZoneInfo("Asia/Manila")
    except Exception:
        from pytz import timezone
        tz = timezone("Asia/Manila")
    now = datetime.now(tz)
    if isinstance(interval, str) and interval.endswith("m"):
        try:
            mins = int(interval[:-1])
            minute = now.minute
            remainder = minute % mins
            if remainder == 0 and now.second == 0 and now.microsecond == 0:
                next_ts = now
            else:
                delta_min = mins - remainder
                next_ts = (now.replace(second=0, microsecond=0) + timedelta(minutes=delta_min))
            return pd.Timestamp(next_ts).tz_localize(tz) if getattr(pd.Timestamp(next_ts), "tzinfo", None) is None else pd.Timestamp(next_ts).tz_convert(tz)
        except Exception:
            return pd.Timestamp(now).tz_localize(tz)
    if interval == "1d":
        next_day = (now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
        return pd.Timestamp(next_day).tz_localize(tz)
    next_hour = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
    return pd.Timestamp(next_hour).tz_localize(tz)

def interval_to_timedelta(interval):
    if isinstance(interval, str):
        iv = interval.strip().lower()
        if iv.endswith('m'):
            try:
                mins = int(iv[:-1])
                return timedelta(minutes=mins)
            except Exception:
                return timedelta(hours=1)
        if iv.endswith('h'):
            try:
                hrs = int(iv[:-1])
                return timedelta(hours=hrs)
            except Exception:
                return timedelta(hours=1)
        if iv.endswith('d'):
            try:
                days = int(iv[:-1])
                return timedelta(days=days)
            except Exception:
                return timedelta(days=1)
    return timedelta(hours=1)

def map_label_to_suggestion(label):
    l = (label or "").strip().lower()
    if l == "up":
        return "Buy"
    if l == "down":
        return "Sell"
    return "Hold"

def canonical_label(raw):
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip().lower()
    if s in ('up','buy','long','bull','increase','rise'):
        return 'up'
    if s in ('down','sell','short','bear','decrease','fall','drop'):
        return 'down'
    if s in ('neutral','hold','no change','flat','unchanged','no reliable consensus','none','n/a','unknown'):
        return 'neutral'
    if 'up' in s and 'down' not in s:
        return 'up'
    if 'down' in s and 'up' not in s:
        return 'down'
    if 'hold' in s or 'neutral' in s or 'no reliable' in s:
        return 'neutral'
    return None

def label_from_model(mod, X_for_proba=None):
    if hasattr(mod, "predict_proba") and X_for_proba is not None:
        try:
            probs = mod.predict_proba(X_for_proba)
            probs = np.asarray(probs)
            if probs.ndim == 2:
                p = probs[0]
            else:
                p = probs.ravel()
            top_idx = int(np.argmax(p))
            top_prob = float(p[top_idx])
            if top_prob >= prob_threshold:
                lab = label_map[top_idx] if top_idx < len(label_map) else str(top_idx)
            else:
                neutral_idx = next((i for i, l in enumerate(label_map) if l.lower() == "neutral"), None)
                lab = label_map[neutral_idx] if neutral_idx is not None else (label_map[top_idx] if top_idx < len(label_map) else str(top_idx))
            return lab, top_prob, p.tolist()
        except Exception:
            pass

    try:
        pred = mod.predict(X_for_proba if X_for_proba is not None else None)
    except Exception:
        try:
            pred = mod.predict(np.asarray(X_for_proba)) if X_for_proba is not None else mod.predict(None)
        except Exception as e:
            return ("unknown", 0.0, str(e))

    pred = np.asarray(pred)
    if pred.ndim == 2 and pred.shape[0] == 1 and pred.shape[1] >= 2:
        p = pred.ravel()
        top_idx = int(np.argmax(p))
        top_prob = float(p[top_idx])
        if top_prob >= prob_threshold:
            lab = label_map[top_idx] if top_idx < len(label_map) else str(top_idx)
        else:
            neutral_idx = next((i for i, l in enumerate(label_map) if l.lower() == "neutral"), None)
            lab = label_map[neutral_idx] if neutral_idx is not None else (label_map[top_idx] if top_idx < len(label_map) else str(top_idx))
        return lab, top_prob, p.tolist()
    try:
        v = int(np.ravel(pred)[0])
        lab = label_map[v] if v < len(label_map) else str(v)
        return lab, 1.0, int(v)
    except Exception:
        return (str(pred), 0.0, pred.tolist() if hasattr(pred, "tolist") else str(pred))

def load_history():
    try:
        if os.path.exists(history_file):
            df = pd.read_csv(history_file)
            for c in ["predicted_at","fetched_last_ts","target_time","checked_at"]:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors='coerce')
            return df
    except Exception:
        pass
    return pd.DataFrame()

def save_history(df):
    save_history_atomic(df, history_file)

def price_at_or_before(series: pd.Series, when_ts: pd.Timestamp):
    if series is None or series.empty:
        return float("nan")
    try:
        s = series.copy()
        s.index = pd.to_datetime(s.index)
        if getattr(s.index, "tz", None) is None:
            s.index = s.index.tz_localize("UTC", ambiguous='infer').tz_convert(ZoneInfo("Asia/Manila"))
        else:
            s.index = s.index.tz_convert(ZoneInfo("Asia/Manila"))
    except Exception as e:
        warn(f"Could not localize/convert series index to Manila TZ for price lookup: {e}")
        s = series.copy()
        s.index = pd.to_datetime(s.index)
    if not s.index.is_monotonic_increasing:
        s = s.sort_index()
    when_ts = ensure_timestamp_in_manila(when_ts)
    val = s.asof(when_ts)
    if pd.notna(val):
        return float(val)
    if when_ts < s.index[0]:
        first_valid = s.dropna()
        return float(first_valid.iloc[0]) if not first_valid.empty else float("nan")
    return float("nan")

def append_prediction_with_dedup(history, new_row, history_file=None, save_history_func=None):
    import pandas as pd

    if history is None or (hasattr(history, 'empty') and history.empty):
        out = pd.DataFrame([new_row])
    else:
        if 'ticker' not in history.columns:
            history['ticker'] = pd.NA
        if 'interval' not in history.columns:
            history['interval'] = pd.NA
        if 'target_time' not in history.columns:
            history['target_time'] = pd.NA

        def _norm_ts(x):
            try:
                return str(ensure_timestamp_in_manila(x))
            except Exception:
                return str(pd.to_datetime(x, errors='coerce'))

        new_tgt_norm = _norm_ts(new_row.get('target_time'))
        hist_ticker = history['ticker'].astype(str).str.upper()
        hist_interval = history['interval'].astype(str)
        hist_target_norm = history['target_time'].apply(lambda x: _norm_ts(x) if pd.notna(x) else "")

        new_key = (str(new_row.get('ticker','')).upper(), str(new_row.get('interval','')), str(new_tgt_norm))
        mask_same = (hist_ticker == new_key[0]) & (hist_interval == new_key[1]) & (hist_target_norm == new_key[2])

        if mask_same.any():
            history = history.loc[~mask_same].reset_index(drop=True)

        missing_cols = set(new_row.keys()) - set(history.columns)
        for mc in missing_cols:
            history[mc] = pd.NA

        out = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True, sort=False)

    try:
        if save_history_func is not None:
            save_history_func(out)
        elif history_file is not None:
            out.to_csv(history_file, index=False)
    except Exception:
        pass

    return out

def evaluate_history(history_df, aligned_close_df, current_fetched_ts):
    import pandas as pd
    import numpy as np
    from zoneinfo import ZoneInfo

    if history_df is None or history_df.empty:
        return history_df

    if 'evaluated' not in history_df.columns:
        history_df['evaluated'] = False

    for c in ['predicted_at','fetched_last_ts','target_time','checked_at']:
        if c in history_df.columns:
            history_df[c] = pd.to_datetime(history_df[c], errors='coerce')

    ac = aligned_close_df.copy()
    ac.index = pd.to_datetime(ac.index)
    if getattr(ac.index, 'tz', None) is None:
        ac.index = ac.index.tz_localize('UTC').tz_convert(ZoneInfo('Asia/Manila'))
    else:
        ac.index = ac.index.tz_convert(ZoneInfo('Asia/Manila'))
    ac = ac.sort_index()
    ac.columns = [str(c).upper() for c in ac.columns]

    try:
        now_ts = ensure_timestamp_in_manila(current_fetched_ts)
    except Exception:
        now_ts = ensure_timestamp_in_manila(pd.Timestamp.now())

    history_df['_target_norm'] = history_df['target_time'].apply(lambda x: ensure_timestamp_in_manila(x) if pd.notna(x) else pd.NaT)
    history_df['_predicted_at_norm'] = history_df['predicted_at'].apply(lambda x: ensure_timestamp_in_manila(x) if pd.notna(x) else pd.NaT)

    default_iv_td = pd.Timedelta(hours=1)
    iv_td = default_iv_td
    if 'interval' in history_df.columns and not history_df['interval'].isna().all():
        try:
            sample_iv = history_df['interval'].dropna().iloc[0]
            iv_td = interval_to_timedelta(sample_iv)
        except Exception:
            iv_td = default_iv_td

    safety_buffer = pd.Timedelta(seconds=max(1, int(iv_td.total_seconds() * 0.05)))
    mask_ready = (~history_df['evaluated'].astype(bool)) & history_df['_target_norm'].notna() & (history_df['_target_norm'] < now_ts)
    recent_pred_mask = history_df['_predicted_at_norm'].notna() & (history_df['_predicted_at_norm'] > (now_ts - safety_buffer))
    mask_ready = mask_ready & (~recent_pred_mask)

    for idx in history_df[mask_ready].index:
        try:
            row = history_df.loc[idx]
            t_target = history_df.at[idx, '_target_norm']
            tk = str(row.get('ticker', '')).upper().strip()
            history_df.at[idx, 'checked_at'] = pd.Timestamp.now(tz=ZoneInfo('Asia/Manila'))

            if not tk or tk not in ac.columns:
                history_df.at[idx, 'eval_error'] = 'invalid_ticker'
                continue

            series = ac[tk].dropna()
            if series.empty:
                history_df.at[idx, 'eval_error'] = 'no_price_data'
                continue

            pred_label = canonical_label(row.get('predicted_label'))
            if pred_label is None:
                history_df.at[idx, 'eval_error'] = f'ambiguous_label:{row.get("predicted_label")}'
                continue

            pred_price_val = None
            if pd.notna(row.get('pred_price')):
                pred_price_val = float(row.get('pred_price'))
            if pred_price_val is None:
                pred_time_candidate = history_df.at[idx, '_predicted_at_norm'] or row.get('fetched_last_ts') or (t_target - pd.Timedelta(seconds=1))
                pred_price_val = price_at_or_before(series, pred_time_candidate)

            actual_price_val = price_at_or_before(series, t_target)

            if pd.isna(pred_price_val) or pd.isna(actual_price_val) or pred_price_val == 0:
                history_df.at[idx, 'eval_error'] = 'missing_or_zero_price'
                continue

            history_df.at[idx, 'pred_price'] = float(pred_price_val)
            history_df.at[idx, 'actual_price'] = float(actual_price_val)
            pct = (actual_price_val - pred_price_val) / pred_price_val
            history_df.at[idx, 'pct_change'] = pct

            iv_seconds = iv_td.total_seconds()
            bars_in_24h = max(1, int(24*3600 / max(1, iv_seconds)))
            rolling_vol = series.pct_change().rolling(window=bars_in_24h, min_periods=1).std()
            vol_ref_time = row.get('fetched_last_ts') or history_df.at[idx, '_predicted_at_norm'] or (t_target - pd.Timedelta(seconds=1))
            vol_at_prediction = rolling_vol.asof(vol_ref_time)
            thr = float(vol_at_prediction)*0.5 if pd.notna(vol_at_prediction) and vol_at_prediction != 0 else 0.002
            history_df.at[idx, 'neutral_threshold_used'] = thr

            if pct > thr: actual_movement_category = 'up'
            elif pct < -thr: actual_movement_category = 'down'
            else: actual_movement_category = 'neutral'

            history_df.at[idx, 'correct'] = (pred_label == actual_movement_category)
            history_df.at[idx, 'evaluated'] = True
            history_df.at[idx, 'eval_error'] = None
        except Exception as e:
            history_df.at[idx, 'eval_error'] = f'exception:{str(e)}'
            continue

    return history_df.drop(columns=['_target_norm','_predicted_at_norm'])

tab1, tab2, tab3, tab4 = st.tabs(["Live Market View", "Predictions", "Detailed Analysis", "History Predictions"])

# ------------------------------
# Run pipeline
# ------------------------------
if run_button:
    with st.spinner("Downloading latest market data (transient)..."):
        needed = {ticker, "SPY", "QQQ", "NVDA"}
        raw = {}
        for t in needed:
            try:
                df = download_price_cached(t, interval=interval, period=period)
                if df is None or df.empty:
                    continue
                df = df.dropna(how="all")
                raw[t] = df
            except Exception as e:
                warn(f"Download failed for {t}: {e}")

    if ticker not in raw:
        st.error(f"No data for {ticker}. Try increasing period or check connectivity.")
        st.stop()

    base_idx = raw[ticker].index
    aligned_close = pd.DataFrame(index=base_idx)
    for t, df in raw.items():
        aligned_close[t] = df.reindex(base_idx)["Close"]

    fetched_last_ts = base_idx[-1]
    fetched_last_ts_manila = ensure_timestamp_in_manila(fetched_last_ts)
    real_next_time = compute_real_next_time_now(interval)
    try:
        tz_manila = ZoneInfo("Asia/Manila")
    except Exception:
        from pytz import timezone
        tz_manila = timezone("Asia/Manila")
    now_manila = datetime.now(tz_manila)
    try:
        age_secs = (now_manila - fetched_last_ts_manila).total_seconds()
    except Exception:
        age_secs = (now_manila.replace(tzinfo=None) - pd.to_datetime(fetched_last_ts_manila).replace(tzinfo=None)).total_seconds()
    interval_seconds = 3600 if interval == "60m" else 86400
    tolerance = interval_seconds * 1.5
    data_is_stale = age_secs > tolerance

    # ----------- FIX: DO NOT instantly evaluate after prediction, only evaluate past history -----------
    try:
        history = load_history()
        history = evaluate_history(history, aligned_close, fetched_last_ts_manila)
        save_history(history)
    except Exception as e:
        warn(f"History evaluation failed: {e}")

    def build_features(aligned_close_df, raw_dict, tgt):
        aligned_ffill = aligned_close_df.ffill()
        price = aligned_ffill[tgt]
        feat = pd.DataFrame(index=price.index)
        for lag in [1,3,6,12,24]:
            feat[f"ret_{lag}h"] = price.pct_change(lag)
        for w in [6,12,24]:
            feat[f"vol_{w}h"] = price.pct_change().rolling(w).std()
        try:
            close_series = price.squeeze() if isinstance(price, pd.DataFrame) else price
            feat["rsi_14"] = ta.momentum.RSIIndicator(close_series, window=14).rsi()
            macd = ta.trend.MACD(close_series)
            feat["macd"] = macd.macd()
            feat["macd_signal"] = macd.macd_signal()
        except Exception:
            feat["rsi_14"] = np.nan
            feat["macd"] = np.nan
            feat["macd_signal"] = np.nan
        for w in [5,10,20]:
            feat[f"sma_{w}"] = price.rolling(w).mean()
            feat[f"ema_{w}"] = price.ewm(span=w, adjust=False).mean()
        if tgt in raw_dict and "Volume" in raw_dict[tgt].columns:
            vol = raw_dict[tgt].reindex(price.index)["Volume"].ffill()
            feat["vol_change_1h"] = vol.pct_change()
            feat["vol_ma_24h"] = vol.rolling(24).mean()
        for asset in ["SPY","QQQ","NVDA"]:
            if asset in aligned_ffill.columns:
                feat[f"{asset}_ret_1h"] = aligned_ffill[asset].pct_change()
        feat["hour"] = feat.index.hour
        feat["day_of_week"] = feat.index.dayofweek
        feat = feat.dropna()
        feat["datetime"] = feat.index
        feat["ticker"] = tgt
        return feat.reset_index(drop=True)

    feat_df = build_features(aligned_close, raw, ticker)
    if feat_df.empty:
        st.error("Not enough history to compute features for this ticker; increase period.")
        st.stop()

    # ... rest of the code continues unchanged ...
    # The rest is as supplied above (tabs, model loading, prediction, evaluation, history, etc.)
    with tab1:
        st.subheader(f"Live Market View — {ticker}")
        st.write(f"Fetched latest timestamp (converted to Manila): **{fetched_last_ts_manila}**")
        st.write(f"Real-time clock now (Asia/Manila): **{now_manila.strftime('%Y-%m-%d %H:%M:%S %Z')}**")
        st.write(f"Real next-interval target (clock-based): **{real_next_time}**")
        if data_is_stale:
            st.caption("Note: fetched market data is stale relative to the real-time clock and may lag real-time.")

        intraday_df = pd.DataFrame()
        try:
            with st.spinner("Fetching intraday ticks (1m granularity, cached)..."):
                intraday_df = download_price_cached(ticker, interval="1m", period="1d", use_tk_history=True)
            if intraday_df is None or intraday_df.empty:
                intraday_df = raw.get(ticker, pd.DataFrame()).copy()
        except Exception:
            intraday_df = raw.get(ticker, pd.DataFrame()).copy()
        chart_df = intraday_df.copy()
        if chart_df is None or chart_df.empty:
            st.info("Intraday tick data not available to draw the realtime line chart.")
        else:
            try:
                chart_df.index = pd.to_datetime(chart_df.index)
            except Exception:
                pass
            price_series = chart_df['Close'].ffill().dropna()
            if price_series.empty:
                st.info("No intraday close prices available.")
            else:
                prev_close = None
                try:
                    hist2d = download_price_cached(ticker, interval="1d", period="2d")
                    if hist2d is not None and len(hist2d) >= 2:
                        prev_close = float(hist2d['Close'].iloc[-2])
                    else:
                        prev_close = float(price_series.iloc[0])
                except Exception:
                    try:
                        prev_close = float(price_series.iloc[0])
                    except Exception:
                        prev_close = None
                current_price = float(price_series.iloc[-1])
                current_time = price_series.index[-1]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=price_series.index,
                    y=price_series.values,
                    mode='lines',
                    name=f"{ticker} Price",
                    line=dict(width=2, color="#00E5FF")
                ))
                try:
                    fig.add_hline(
                        y=current_price,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Current: {current_price:.2f}",
                        annotation_position="top left",
                        annotation=dict(font=dict(color="white"))
                    )
                except Exception:
                    pass
                if prev_close is not None:
                    try:
                        fig.add_hline(
                            y=prev_close,
                            line_dash="dash",
                            line_color="white",
                            annotation_text=f"Prev Close: {prev_close:.2f}",
                            annotation_position="bottom left",
                            annotation=dict(font=dict(color="white"))
                        )
                    except Exception:
                        pass
                fig.add_trace(go.Scatter(
                    x=[current_time],
                    y=[current_price],
                    mode='markers',
                    marker=dict(color='red', size=8),
                    name='Current Price'
                ))
                fig.update_layout(
                    plot_bgcolor="#0b1020",
                    paper_bgcolor="#0b1020",
                    font_color="white",
                    height=520,
                    hovermode="x unified",
                    margin=dict(l=10, r=10, t=30, b=30),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", gridwidth=1, zeroline=False, color="white")
                fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", gridwidth=1, zeroline=False, color="white")
                fig.update_xaxes(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=60, label="1h", step="minute", stepmode="backward"),
                            dict(count=360, label="6h", step="minute", stepmode="backward"),
                            dict(count=1, label="1d", step="day", stepmode="backward"),
                            dict(step="all", label="All")
                        ]),
                        bgcolor="#111827",
                        activecolor="#0ea5a4",
                    ),
                    rangeslider=dict(visible=False)
                )
                st.plotly_chart(fig, use_container_width=True)

    with st.spinner("Loading models (transient)..."):
        file_map = {"cnn": cnn_path, "lstm": lstm_path, "xgb": xgb_path, "meta": meta_path}
        loaded = {}
        for name in models_to_run:
            path = file_map.get(name)
            if not path:
                st.warning(f"No path set for {name}; skipping.")
                continue
            if not os.path.exists(path):
                st.warning(f"Model file missing: {path}; skipping {name}.")
                continue
            mod = load_model_safe(path)
            if mod is not None:
                loaded[name] = mod

    if not loaded:
        st.error("No models loaded. Place .pkl files next to app.py or correct paths.")
        st.stop()

    last_row = feat_df.drop(columns=["datetime","ticker"], errors='ignore').iloc[-1:].replace([np.inf,-np.inf], np.nan).fillna(0.0)
    seq_df = feat_df.drop(columns=["datetime","ticker"], errors='ignore').replace([np.inf,-np.inf], np.nan).fillna(0.0)

    results = []
    for name, mod in loaded.items():
        try:
            if name in ("cnn","lstm"):
                expected = get_feature_names(mod)
                seq_for_model = seq_df.copy()
                if expected is not None:
                    seq_for_model = align_seq_df_to_names(seq_for_model, expected)
                else:
                    n_feat = get_n_features(mod)
                    if n_feat is not None:
                        if seq_for_model.shape[1] < n_feat:
                            for i in range(n_feat - seq_for_model.shape[1]):
                                seq_for_model[f"_pad_{i}"] = 0.0
                        elif seq_for_model.shape[1] > n_feat:
                            seq_for_model = seq_for_model.iloc[:, :n_feat]
                try:
                    scaler = MinMaxScaler().fit(seq_for_model)
                    seq_scaled = pd.DataFrame(scaler.transform(seq_for_model), columns=seq_for_model.columns)
                except Exception:
                    seq_scaled = seq_for_model.copy()
                arr = seq_scaled.values
                if arr.shape[0] >= sequence_length:
                    slice_arr = arr[-sequence_length:, :]
                else:
                    pad = np.zeros((sequence_length - arr.shape[0], arr.shape[1]))
                    slice_arr = np.vstack([pad, arr])
                X_in = np.expand_dims(slice_arr, axis=0)
                try:
                    raw_pred = mod.predict(X_in)
                except Exception:
                    try:
                        raw_pred = mod.predict(X_in.reshape((1,-1)))
                    except Exception:
                        raw_pred = None
                lab, conf, rawinfo = label_from_model(mod, X_in)
                lab_str = str(lab).strip()
                lab_canon = lab_str.lower()
                if lab_canon not in ("up","down","neutral"):
                    try:
                        if isinstance(rawinfo, (int, np.integer)):
                            lab_idx = int(rawinfo)
                            if 0 <= lab_idx < len(label_map):
                                lab_canon = label_map[lab_idx].lower()
                    except Exception:
                        pass
                lab_display = lab_canon.capitalize() if lab_canon in ("up","down","neutral") else lab_str.capitalize()
                results.append({
                    "model": name,
                    "label": lab_display,
                    "confidence": round(float(conf),4),
                    "suggestion": map_label_to_suggestion(lab_canon),
                    "raw": rawinfo
                })
            else:
                expected = get_feature_names(mod)
                n_feat = get_n_features(mod)
                if expected is not None:
                    hist_for_scaler = seq_df.reindex(columns=expected).fillna(0.0)
                    try:
                        scaler = MinMaxScaler().fit(hist_for_scaler)
                    except Exception:
                        scaler = None
                    X_use = align_tabular_row_to_names(last_row, expected)
                    if scaler is not None:
                        try:
                            X_scaled_np = scaler.transform(X_use)
                            X_scaled_df = pd.DataFrame(X_scaled_np, columns=expected)
                        except Exception:
                            X_scaled_df = X_use.copy()
                    else:
                        X_scaled_df = X_use.copy()
                    try:
                        lab, conf, rawinfo = label_from_model(mod, X_scaled_df)
                        lab_str = str(lab).strip()
                        lab_canon = lab_str.lower()
                        if lab_canon not in ("up","down","neutral"):
                            try:
                                if isinstance(rawinfo, (int, np.integer)):
                                    lab_idx = int(rawinfo)
                                    if 0 <= lab_idx < len(label_map):
                                        lab_canon = label_map[lab_idx].lower()
                            except Exception:
                                pass
                        lab_display = lab_canon.capitalize() if lab_canon in ("up","down","neutral") else lab_str.capitalize()
                        results.append({
                            "model": name,
                            "label": lab_display,
                            "confidence": round(float(conf),4),
                            "suggestion": map_label_to_suggestion(lab_canon),
                            "raw": rawinfo
                        })
                    except Exception as e:
                        msg = str(e)
                        provided = set(last_row.columns)
                        missing = sorted(list(set(expected) - provided))
                        unseen = sorted(list(provided - set(expected)))
                        st.error(f"{name} prediction failed: {msg}")
                        st.info(f"Missing features (sample up to 10): {missing[:10]}")
                        if unseen:
                            st.info(f"Provided features unseen by model (sample up to 10): {unseen[:10]}")
                        results.append({"model":name,"label":"ERROR","confidence":0.0,"suggestion":"ERROR","raw":msg})
                        continue
                elif n_feat is not None:
                    X_arr = last_row.values
                    if X_arr.shape[1] < n_feat:
                        pad = np.zeros((1, n_feat - X_arr.shape[1]))
                        X_arr = np.hstack([X_arr, pad])
                    elif X_arr.shape[1] > n_feat:
                        X_arr = X_arr[:, :n_feat]
                    hist = seq_df.values
                    if hist.shape[1] >= n_feat:
                        hist_slice = hist[:, :n_feat]
                    else:
                        hist_slice = np.hstack([hist, np.zeros((hist.shape[0], n_feat - hist.shape[1]))])
                    try:
                        scaler = MinMaxScaler().fit(hist_slice)
                        X_scaled_arr = scaler.transform(X_arr)
                    except Exception:
                        X_scaled_arr = X_arr
                    try:
                        lab, conf, rawinfo = label_from_model(mod, X_scaled_arr)
                        lab_str = str(lab).strip()
                        lab_canon = lab_str.lower()
                        if lab_canon not in ("up","down","neutral"):
                            try:
                                if isinstance(rawinfo, (int, np.integer)):
                                    lab_idx = int(rawinfo)
                                    if 0 <= lab_idx < len(label_map):
                                        lab_canon = label_map[lab_idx].lower()
                            except Exception:
                                pass
                        lab_display = lab_canon.capitalize() if lab_canon in ("up","down","neutral") else lab_str.capitalize()
                        results.append({
                            "model": name,
                            "label": lab_display,
                            "confidence": round(float(conf),4),
                            "suggestion": map_label_to_suggestion(lab_canon),
                            "raw": rawinfo
                        })
                    except Exception as e:
                        st.error(f"{name} prediction failed: {e}")
                        results.append({"model":name,"label":"ERROR","confidence":0.0,"suggestion":"ERROR","raw":str(e)})
                        continue
                else:
                    st.error(f"Model {name} lacks feature info; cannot align.")
                    results.append({"model":name,"label":"ERROR","confidence":0.0,"suggestion":"ERROR","raw":"no feature info"})
                    continue
        except Exception as e:
            st.error(f"Unhandled error predicting with {name}: {e}")
            results.append({"model":name,"label":"ERROR","confidence":0.0,"suggestion":"ERROR","raw":str(e)})

    with tab2:
        st.subheader("Predictions — next interval (real-time)")
        st.write(f"Ticker: **{ticker}**   •   Fetched latest (Manila): **{fetched_last_ts_manila}**")
        st.write(f"Real-time clock now (Asia/Manila): **{now_manila.strftime('%Y-%m-%d %H:%M:%S %Z')}**")
        st.write(f"Real next-interval target (clock-based): **{real_next_time}**")
        if data_is_stale:
            st.caption("Note: fetched market data is stale relative to the real-time clock and may lag real-time.")
        if results:
            out = pd.DataFrame(results)[["model", "label", "confidence", "suggestion", "raw"]]
            try:
                st.dataframe(out)
            except Exception:
                st.dataframe(out)
        else:
            st.warning("No predictions produced.")
        con_label = None; con_conf = 0.0; con_src = "ensemble"
        if "meta" in loaded:
            try:
                meta_mod = loaded["meta"]
                expected = get_feature_names(meta_mod)
                n_feat = get_n_features(meta_mod)
                if expected is not None:
                    X_meta = align_tabular_row_to_names(last_row, expected)
                    try:
                        scaler = MinMaxScaler().fit(seq_df.reindex(columns=expected).fillna(0.0))
                        X_scaled_meta = pd.DataFrame(scaler.transform(X_meta), columns=expected)
                        lab, conf, rawinfo = label_from_model(meta_mod, X_scaled_meta)
                    except Exception:
                        lab, conf, rawinfo = label_from_model(meta_mod, X_meta)
                elif n_feat is not None:
                    X_arr = last_row.values
                    if X_arr.shape[1] < n_feat:
                        pad = np.zeros((1, n_feat - X_arr.shape[1]))
                        X_arr = np.hstack([X_arr, pad])
                    elif X_arr.shape[1] > n_feat:
                        X_arr = X_arr[:, :n_feat]
                    hist = seq_df.values
                    if hist.shape[1] >= n_feat:
                        hist_slice = hist[:, :n_feat]
                    else:
                        hist_slice = np.hstack([hist, np.zeros((hist.shape[0], n_feat - hist.shape[1]))])
                    try:
                        scaler = MinMaxScaler().fit(hist_slice)
                        X_scaled = scaler.transform(X_arr)
                    except Exception:
                        X_scaled = X_arr
                    lab, conf, rawinfo = label_from_model(meta_mod, X_scaled)
                else:
                    raise RuntimeError("meta has no feature info")
                meta_label = str(lab).strip().lower()
                if meta_label in ("up", "down", "neutral"):
                    con_label = meta_label.capitalize()
                    con_conf = float(conf)
                    con_src = "meta"
                else:
                    con_label = None
            except Exception as e:
                st.warning(f"Meta failed to produce conclusive output: {e}. Falling back to ensemble.")
                con_label = None
        if con_label is None:
            valid_rows = [r for r in results if r.get("label") not in ("ERROR", "Unknown", "unknown")]
            if valid_rows:
                votes = {}
                for r in valid_rows:
                    lbl = r["label"].lower() if isinstance(r.get("label"), str) else ""
                    votes.setdefault(lbl, []).append(r.get("confidence", 0.0))
                best_label = None; best_count = -1; best_conf = -1.0
                for lbl, confs in votes.items():
                    cnt = len(confs)
                    avgc = float(np.mean(confs)) if confs else 0.0
                    if (cnt > best_count) or (cnt == best_count and avgc > best_conf):
                        best_label = lbl; best_count = cnt; best_conf = avgc
                if best_label is not None and best_label in ("up", "down", "neutral"):
                    con_label = best_label.capitalize()
                    con_conf = float(best_conf)
                    con_src = "ensemble"
                else:
                    con_label = "No reliable consensus"
                    con_conf = 0.0
                    con_src = "none"
            else:
                con_label = "No reliable consensus"
                con_conf = 0.0
                con_src = "none"
        if con_label:
            iv = interval if isinstance(interval, str) else str(interval)
            if iv == "60m":
                period_word = "hour"
            elif iv == "1d":
                period_word = "day"
            else:
                period_word = iv
            if con_label.lower() == "up":
                verb = "go up"
                bg = COLOR_BUY_BG
            elif con_label.lower() == "down":
                verb = "go down"
                bg = COLOR_SELL_BG
            else:
                verb = "be neutral"
                bg = COLOR_HOLD_BG
            sentence = f"{ticker} will likely {verb} the next {period_word}."
            html = f"""
            <div style="padding:14px;border-radius:8px;background:{bg};color:{COLOR_TEXT}">
              <h3 style="margin:0;">{sentence}</h3>
              <p style="margin:4px 0 0 0;"><strong>Suggestion:</strong> {map_label_to_suggestion(con_label.lower())}</p>
              <p style="margin:4px 0 0 0;"><strong>Source:</strong> {con_src}</p>
              <p style="margin:4px 0 0 0;"><strong>Confidence:</strong> {round(con_conf,4)}</p>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)
        try:
            history = load_history()
        except Exception:
            history = pd.DataFrame()
        try:
            pred_price = None
            thr_used = None
            if ticker in aligned_close.columns:
                try:
                    pred_price = price_at_or_before(aligned_close[ticker], fetched_last_ts_manila)
                except Exception as e:
                    warn(f"Could not get pred_price for new history row: {e}")
                    pred_price = None
                try:
                    series_for_thr = aligned_close[ticker].dropna()
                    if not series_for_thr.empty:
                        rolling_vol_now = series_for_thr.pct_change().rolling(window=24, min_periods=1).std()
                        vol_at_pred = rolling_vol_now.asof(fetched_last_ts_manila)
                        if pd.isna(vol_at_pred) or vol_at_pred == 0:
                            thr_used = 0.002
                        else:
                            thr_used = float(vol_at_pred) * 0.5
                    else:
                        thr_used = 0.002
                except Exception:
                    thr_used = 0.002
        except Exception:
            pred_price = None
            thr_used = 0.002
        try:
            delta = interval_to_timedelta(interval)
            # Use the clock-based next interval (safer), but ensure it is strictly in the future
            real_next = compute_real_next_time_now(interval)
            now_manila = ensure_timestamp_in_manila(pd.Timestamp.now())
            target_time_from_fetch = real_next
            # If real_next is already <= now (rare), then push forward by deltas until it is in the future
            while target_time_from_fetch <= now_manila:
                target_time_from_fetch = target_time_from_fetch + delta
        except Exception:
            target_time_from_fetch = compute_real_next_time_now(interval)

        try:
            pred_label_canon = canonical_label(con_label)
        except Exception:
            pred_label_canon = canonical_label(str(con_label)) if con_label is not None else None
        new_row = {
            'predicted_at': pd.Timestamp.now(tz_manila),
            'ticker': ticker,
            'interval': interval,
            'predicted_label': con_label if isinstance(con_label, str) else str(con_label),
            'predicted_label_canonical': pred_label_canon,
            'suggestion': map_label_to_suggestion(con_label.lower()) if isinstance(con_label, str) else '',
            'confidence': float(con_conf) if con_conf is not None else None,
            'fetched_last_ts': fetched_last_ts_manila,
            'target_time': target_time_from_fetch,
            'pred_price': pred_price,
            'neutral_threshold_used': float(thr_used) if thr_used is not None else None,
            'evaluated': False,
            'actual_price': None,
            'pct_change': None,
            'correct': None,
            'checked_at': None,
            'eval_error': None
        }
        try:
            if history is None or history.empty:
                history = pd.DataFrame([new_row])
            else:
                missing_cols = set(new_row.keys()) - set(history.columns)
                for mc in missing_cols:
                    history[mc] = pd.NA
                history = append_prediction_with_dedup(history, new_row, history_file=history_file, save_history_func=save_history)

            save_history(history)
        except Exception as e:
            warn(f"Failed to append/save history: {e}")
        try:
            history = load_history()
            if not history.empty:
                hist_t = history[history['ticker'].str.upper() == ticker.upper()].sort_values('predicted_at', ascending=False)
                if not hist_t.empty:
                    st.subheader('Prediction history — this ticker')
                    cols = ['predicted_at','target_time','predicted_label','suggestion','confidence','pred_price','actual_price','pct_change','correct','evaluated','checked_at','eval_error','neutral_threshold_used']
                    available = [c for c in cols if c in hist_t.columns]
                    st.dataframe(hist_t[available].head(50))
        except Exception:
            pass
        if results:
            st.download_button("Download per-model predictions CSV", data=pd.DataFrame(results).to_csv(index=False).encode("utf-8"), file_name="predictions_next_interval.csv")

    # ... rest of code continues unchanged ...
# (History Predictions tab and other tabs remain as in previous version, with evaluation logic now fixed)
    # ------------------------------
    # Detailed Analysis tab
    # ------------------------------
    with tab3:
        st.subheader("Detailed Analysis — Defense-Friendly")

        if not results:
            st.info("Run a prediction to see detailed analysis and model outputs.")
        else:
            # --- 1) Confidence & Probability Breakdown ---
            st.markdown("### 1) Confidence & Probability Breakdown")
            try:
                class_keys = ['up','neutral','down']
                agg = {k:0.0 for k in class_keys}
                count = 0
                for r in results:
                    raw = r.get('raw')
                    if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) == len(label_map):
                        for i,p in enumerate(raw):
                            lbl = label_map[i].lower()
                            if lbl in agg:
                                agg[lbl] += float(p)
                        count += 1
                    else:
                        lbl = r.get('label','').lower()
                        try:
                            agg[lbl] += float(r.get('confidence',0.0))
                        except Exception:
                            pass
                        count += 1
                if count > 0:
                    probs = {k: (agg[k] / count) for k in class_keys}
                else:
                    probs = {k:0.0 for k in class_keys}

                pct_display = {k: f"{probs[k]*100:.1f}%" for k in class_keys}
                st.write(f"Uptrend: {pct_display['up']} — Neutral: {pct_display['neutral']} — Downtrend: {pct_display['down']}")

                try:
                    fig_prob = go.Figure(go.Bar(
                        x=[probs['up']*100, probs['neutral']*100, probs['down']*100],
                        y=['Up','Neutral','Down'],
                        orientation='h'
                    ))
                    fig_prob.update_layout(height=250, xaxis_title='Probability (%)')
                    st.plotly_chart(fig_prob, use_container_width=True)
                except Exception:
                    pass

                try:
                    # For the gauge, we can't use a dynamic threshold directly,
                    # but we can still show the overall confidence.
                    val = float(con_conf) * 100 if con_conf is not None else 0.0
                    fig_gauge = go.Figure(go.Indicator(mode='gauge+number', value=val,
                                                      gauge={'axis':{'range':[0,100]}}))
                    fig_gauge.update_layout(height=250, margin={'t':20,'b':20,'l':20,'r':20})
                    st.plotly_chart(fig_gauge, use_container_width=True)
                except Exception:
                    pass
            except Exception as e:
                st.warning(f"Confidence breakdown failed: {e}")

            # --- 2) Model Ensemble Outputs ---
            st.markdown("### 2) Model Ensemble Outputs")
            try:
                df_models = pd.DataFrame([{'Model':r['model'].upper(), 'Prediction': r['label'].upper(), 'Confidence': f"{r['confidence']*100:.1f}%"} for r in results])
                st.table(df_models)
            except Exception:
                st.write(results)

            # --- 3) Key Technical Indicators Used ---
            st.markdown("### 3) Key Technical Indicators (latest values)")
            try:
                last_idx = chart_df.index[-1] if 'chart_df' in locals() and not chart_df.empty else None
                indicators = {}
                try:
                    indicators['Close'] = float(chart_df['Close'].iloc[-1]) if last_idx is not None else None
                except Exception:
                    indicators['Close'] = None
                try:
                    indicators['RSI(14)'] = float(ta.momentum.RSIIndicator(chart_df['Close'], window=14).rsi().iloc[-1]) if last_idx is not None else None
                except Exception:
                    indicators['RSI(14)'] = None
                try:
                    macd = ta.trend.MACD(chart_df['Close']) if last_idx is not None else None
                    if macd is not None:
                        indicators['MACD'] = float(macd.macd().iloc[-1])
                        indicators['MACD_signal'] = float(macd.macd_signal().iloc[-1])
                except Exception:
                    indicators['MACD'] = indicators.get('MACD', None)
                for k,v in indicators.items():
                    st.write(f"**{k}:** {v}")
            except Exception:
                st.write("Indicator summary not available.")

            # --- 4) Feature importance (if available) ---
            st.markdown("### 4) Feature importance / coefficients (meta or models)")
            try:
                shown = False
                if 'meta' in loaded:
                    mod = loaded['meta']
                    if hasattr(mod, 'feature_importances_'):
                        fi = getattr(mod, 'feature_importances_')
                        fnames = get_feature_names(mod) or [f"f{i}" for i in range(len(fi))]
                        df_fi = pd.DataFrame({'feature':fnames, 'importance':fi}).sort_values('importance', ascending=False)
                        st.dataframe(df_fi.head(50))
                        shown = True
                    elif hasattr(mod, 'coef_'):
                        coefs = np.ravel(getattr(mod, 'coef_'))
                        fnames = get_feature_names(mod) or [f"f{i}" for i in range(len(coefs))]
                        df_coef = pd.DataFrame({'feature':fnames, 'coef':coefs}).sort_values('coef', ascending=False)
                        st.dataframe(df_coef.head(50))
                        shown = True
                if not shown:
                    for name, mod in loaded.items():
                        if hasattr(mod, 'feature_importances_'):
                            fi = getattr(mod, 'feature_importances_')
                            fnames = get_feature_names(mod) or [f"f{i}" for i in range(len(fi))]
                            df_fi = pd.DataFrame({'feature':fnames, 'importance':fi}).sort_values('importance', ascending=False)
                            st.markdown(f"**{name} feature importances**")
                            st.dataframe(df_fi.head(50))
            except Exception:
                st.write("No feature importance available.")

            # --- 5) Model agreement visualization ---
            st.markdown("### 5) Model agreement")
            votes = {}
            for r in results:
                lbl = r.get('label','').lower()
                votes[lbl] = votes.get(lbl, 0) + 1
            vote_df = pd.DataFrame(list(votes.items()), columns=['label','count'])
            if not vote_df.empty:
                try:
                    st.bar_chart(vote_df.set_index('label'))
                except Exception:
                    st.dataframe(vote_df)

            # --- 6) History accuracy summary ---
            st.markdown("### 6) Recent prediction accuracy")
            try:
                history = load_history()
                if not history.empty:
                    evald = history[history['correct'].notna()]
                    if not evald.empty:
                        acc = evald['correct'].mean()
                        st.write(f"Overall accuracy (evaluated rows): {acc:.3%} ({len(evald)} rows)")
                        per = evald.groupby('ticker')['correct'].agg(['mean','count']).sort_values('count', ascending=False)
                        st.dataframe(per)
                    else:
                        st.write("No evaluated history rows yet. Predictions will be evaluated when their target time passes.")
                else:
                    st.write("No history yet.")
            except Exception:
                st.write("Could not compute history accuracy.")

# end if run_button

# ------------------------------
# History Predictions tab (fixed placement)
# ------------------------------
# Place this code after your other tab code, replacing your current "History Predictions" tab

with tab4:
    st.subheader("History: Prediction Evaluation & Accuracy")

    # Load history once at the top
    history = load_history()
    if history is None or history.empty:
        st.info("No prediction history found. Predictions will appear here after running the model.")
        st.stop()

    # Convert datetime columns
    for c in ['predicted_at', 'fetched_last_ts', 'target_time', 'checked_at']:
        if c in history.columns:
            history[c] = pd.to_datetime(history[c], errors='coerce')

    # Buttons for evaluation actions
    col_reval, col_force = st.columns(2)
    with col_reval:
        re_eval_all = st.button("Re-evaluate ALL history", key="re_eval_all")
    with col_force:
        force_pending = st.button("Force re-evaluate pending", key="force_pending")

    # --- Re-evaluate ALL history (downloads all needed price data for all tickers) ---
    if re_eval_all:
        st.info("Re-evaluating all history rows. This may take a while and will download price data for all tickers.")
        # Reset evaluated marker so all rows will be attempted
        history['evaluated'] = False
        for c in ['eval_error', 'checked_at', 'actual_price', 'pct_change', 'correct']:
            if c in history.columns:
                history[c] = pd.NA
        # Gather all tickers/intervals
        groups = history.groupby(['ticker', 'interval'])
        raw_price_map = {}
        # Compute global min/max for download bounds fallback
        try:
            global_min_fetched = history['fetched_last_ts'].min()
        except Exception:
            global_min_fetched = pd.NaT
        try:
            global_max_target = history['target_time'].max()
        except Exception:
            global_max_target = pd.NaT

        for (tk, iv), grp in groups:
            tk = str(tk).strip().upper()
            if not tk: continue
            yf_interval = iv if isinstance(iv, str) else "60m"
            min_fetched = grp['fetched_last_ts'].min() if 'fetched_last_ts' in grp.columns else pd.NaT
            max_target = grp['target_time'].max() if 'target_time' in grp.columns else pd.NaT
            if pd.isna(min_fetched) and pd.notna(global_min_fetched):
                min_fetched = global_min_fetched
            if pd.isna(max_target) and pd.notna(global_max_target):
                max_target = global_max_target
            start = (pd.to_datetime(min_fetched) - pd.Timedelta(days=3)).date() if pd.notna(min_fetched) else (
                (pd.to_datetime(max_target) - pd.Timedelta(days=7)).date() if pd.notna(max_target) else (pd.Timestamp.now() - pd.Timedelta(days=90)).date()
            )
            end = (pd.to_datetime(max_target) + pd.Timedelta(days=2)).date() if pd.notna(max_target) else pd.Timestamp.now().date()
            use_tk_history = isinstance(yf_interval, str) and yf_interval.endswith('m')
            try:
                df = download_price_cached(tk, interval=yf_interval, start=start, end=end, use_tk_history=use_tk_history)
                if df is not None and not df.empty:
                    raw_price_map[tk] = df
            except Exception as e:
                st.warning(f"Download failed for {tk}/{iv}: {e}")
        if not raw_price_map:
            st.error("Could not fetch price data for any tickers in history. Aborting re-evaluation.")
            st.stop()
        all_index = sorted(set().union(*(df.index for df in raw_price_map.values())))
        aligned_close_for_eval = pd.DataFrame(index=all_index)
        for t, df in raw_price_map.items():
            aligned_close_for_eval[t] = df.reindex(all_index)["Close"]
        try:
            eval_ts = datetime.now(ZoneInfo("Asia/Manila"))
        except Exception:
            eval_ts = datetime.utcnow()
        history = evaluate_history(history, aligned_close_for_eval, eval_ts)
        save_history(history)
        st.success("Re-evaluation complete and history saved.")

    # --- Force re-evaluate only pending predictions ---
    if force_pending:
        st.info("Re-evaluating pending predictions only.")
        unevaluated = history[~history['evaluated'].astype(bool)]
        if unevaluated.empty:
            st.info("No pending predictions to evaluate.")
        else:
            raw_price_map = {}
            groups = unevaluated.groupby(['ticker', 'interval'])
            for (tk, iv), grp in groups:
                tk = str(tk).strip().upper()
                yf_interval = iv if isinstance(iv, str) else "60m"
                min_fetched = grp['fetched_last_ts'].min()
                max_target = grp['target_time'].max()
                start = (pd.to_datetime(min_fetched) - pd.Timedelta(days=2)).date() if pd.notna(min_fetched) else (
                    (pd.to_datetime(max_target) - pd.Timedelta(days=5)).date() if pd.notna(max_target) else (pd.Timestamp.now() - pd.Timedelta(days=10)).date()
                )
                end = (pd.to_datetime(max_target) + pd.Timedelta(days=1)).date() if pd.notna(max_target) else pd.Timestamp.now().date()
                try:
                    df = download_price_cached(tk, interval=yf_interval, start=start, end=end)
                    if df is not None and not df.empty:
                        raw_price_map[tk] = df
                except Exception as e:
                    st.warning(f"Download failed for {tk}/{iv}: {e}")
            if raw_price_map:
                all_index = sorted(set().union(*(df.index for df in raw_price_map.values())))
                aligned_close_for_eval = pd.DataFrame(index=all_index)
                for t, df in raw_price_map.items():
                    aligned_close_for_eval[t] = df.reindex(all_index)["Close"]
                try:
                    now_manila = datetime.now(ZoneInfo("Asia/Manila"))
                except Exception:
                    now_manila = datetime.utcnow()
                history = evaluate_history(history, aligned_close_for_eval, now_manila)
                save_history(history)
                st.success("Force evaluation complete and history updated where possible.")
            else:
                st.info("Could not fetch price series for the pending tickers — will re-attempt later.")

    # --- Summary Accuracy Stats (using deduped history) ---
    st.markdown("### Summary Accuracy Stats (Deduped)")

    # Deduplicate history by (ticker, interval, target_time), keep latest prediction
    dedup_cols = ['ticker', 'interval', 'target_time']
    deduped_history = history.sort_values('predicted_at', ascending=False).drop_duplicates(subset=dedup_cols, keep='first')

    total_preds = len(deduped_history)
    evaluated_rows = deduped_history[deduped_history['evaluated'].astype(bool)]
    evaluated_count = len(evaluated_rows)
    correct_count = int(evaluated_rows['correct'].sum()) if evaluated_count > 0 else 0
    accuracy = (correct_count / evaluated_count) if evaluated_count > 0 else 0.0

    cols = st.columns([2, 1])
    with cols[0]:
        st.write(f"**Total Predictions Made:** {total_preds}")
        st.write(f"**Correct Predictions:** {correct_count}")
        st.write(f"**Evaluated:** {evaluated_count}")
        st.write(f"**Accuracy:** {accuracy:.1%}" if evaluated_count > 0 else "**Accuracy:** N/A (no evaluated rows)")
    with cols[1]:
        pending_count = total_preds - evaluated_count
        incorrect = evaluated_count - correct_count
        pie_vals = [correct_count, incorrect, pending_count]
        pie_labels = ["Correct", "Incorrect", "Pending"]
        try:
            fig_donut = go.Figure(go.Pie(labels=pie_labels, values=pie_vals, hole=0.6))
            fig_donut.update_layout(showlegend=True, margin=dict(t=0,b=0,l=0,r=0), height=200)
            st.plotly_chart(fig_donut, use_container_width=True)
        except Exception:
            st.write(f"Correct: {correct_count}  Incorrect: {incorrect}  Pending: {pending_count}")
  # Replace the "Recent Predictions Table" section in your History Predictions tab with this block

    # --- Recent Predictions Table (deduplicated by ticker+interval+target_time, keep latest by predicted_at) ---
    st.markdown("### Recent Predictions Table (Deduped by Ticker/Interval/Target Time)")

    # Deduplicate: keep only the most recent prediction for each (ticker, interval, target_time)
    # Sort by predicted_at descending, then drop duplicates (keep first = most recent)
    dedup_cols = ['ticker', 'interval', 'target_time']
    deduped_history = history.sort_values('predicted_at', ascending=False).drop_duplicates(subset=dedup_cols, keep='first')

    display_rows = []
    for _, row in deduped_history.sort_values('predicted_at', ascending=False).head(200).iterrows():
        pred_at = row.get('predicted_at')
        tk = str(row.get('ticker','')).upper()
        pred_lbl = str(row.get('predicted_label','')).upper() if pd.notna(row.get('predicted_label')) else ""
        actual_move = ""
        correct_mark = "⏳"
        if pd.notna(row.get('actual_price')) and pd.notna(row.get('pred_price')):
            try:
                pct = float(row.get('pct_change'))
                display_thr = float(row.get('neutral_threshold_used')) if pd.notna(row.get('neutral_threshold_used')) else 0.002
                if pct > display_thr:
                    actual_move = "UP"
                elif pct < -display_thr:
                    actual_move = "DOWN"
                else:
                    actual_move = "NEUTRAL"
            except Exception:
                actual_move = ""
        else:
            actual_move = ""
        if pd.notna(row.get('correct')):
            correct_mark = "✅" if bool(row.get('correct')) else "❌"
        else:
            correct_mark = "⏳"
        display_rows.append({
            "Date/Time": pred_at,
            "Ticker": tk,
            "Interval": row.get('interval'),
            "Target Time": row.get('target_time'),
            "Predicted Movement": pred_lbl,
            "Actual Movement": actual_move,
            "Correct?": correct_mark,
            "Eval Error": row.get('eval_error', None)
        })

    df_display = pd.DataFrame(display_rows)
    if not df_display.empty:
        try:
            st.dataframe(df_display)
        except Exception:
            st.write(df_display)
    else:
        st.write("No rows to show.")

    # Download deduped history
    try:
        csv_bytes = deduped_history.to_csv(index=False).encode("utf-8")
        st.download_button("Download deduped history CSV", data=csv_bytes, file_name="predictions_history_deduped.csv")
    except Exception:
        pass
# -----------------------------------------------------------------------------
# Utility: recompute_history_evaluations
# -----------------------------------------------------------------------------
import numpy as np

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


# --- Auto-repair history before showing recent predictions ---
if history is not None and not history.empty:
    try:
        history = recompute_history_evaluations(history, aligned_close_for_eval, save_back=False)
    except Exception as e:
        st.error(f"Repairing history failed: {e}")


