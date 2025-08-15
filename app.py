# app.py - SOLARIS — Stock Price Movement Predictor (Dark intraday line chart + robust history append)
import os
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
    # fallback (keeps usage same elsewhere)
    from pytz import timezone as ZoneInfo

import random, tensorflow as tf

# reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

st.set_page_config(page_title="Solaris Reborn — Real-time Next-Hour", layout="wide")
st.title("SOLARIS — Stock Price Movement Predictor")
st.markdown("_A machine learning–driven hybrid model for short-term market trend forecasting_")

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker (any Yahoo Finance symbol)", "AAPL")
# normalize ticker to uppercase symbol
ticker = ticker.strip().upper()

# Sequence length (restored)
sequence_length = st.sidebar.number_input("Sequence length (LSTM/CNN)", min_value=3, max_value=240, value=24, step=1)

interval = st.sidebar.selectbox("Interval", ["60m", "1d"], index=0)
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

neutral_threshold = st.sidebar.number_input(
    "Neutral threshold (abs % move to call 'Neutral')",
    min_value=0.0, max_value=0.1, value=0.002, step=0.0005, format="%.4f"
)
history_file = st.sidebar.text_input("History CSV file", "predictions_history.csv")

# muted colors (for conclusive box background)
COLOR_BUY_BG = "#d7e9da"   # soft green
COLOR_SELL_BG = "#f0e5e6"  # soft rose
COLOR_HOLD_BG = "#f3f4f6"  # soft gray
COLOR_TEXT = "#111111"

# ------------------------------
# Helpers
# ------------------------------
def safe_load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Failed to load {path}: {e}")
        return None

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
    if inp is not None and isinstance(inp, tuple):
        try:
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

def label_from_model(mod, X_for_proba=None):
    # prefer predict_proba if available
    if hasattr(mod, "predict_proba") and X_for_proba is not None:
        try:
            probs = mod.predict_proba(X_for_proba)
            p = probs[0] if probs.ndim==2 else probs
            top_idx = int(np.argmax(p))
            top_prob = float(p[top_idx])
            if top_prob >= prob_threshold:
                lab = label_map[top_idx] if top_idx < len(label_map) else str(top_idx)
            else:
                neutral_idx = next((i for i,l in enumerate(label_map) if l.lower()=="neutral"), None)
                lab = label_map[neutral_idx] if neutral_idx is not None else (label_map[top_idx] if top_idx < len(label_map) else str(top_idx))
            return lab, top_prob, p.tolist()
        except Exception:
            pass
    # fallback predict
    try:
        pred = mod.predict(X_for_proba if X_for_proba is not None else None)
    except Exception:
        pred = mod.predict(X_for_proba if X_for_proba is not None else None)
    arr = np.array(pred)
    try:
        v = int(arr.ravel()[0])
        lab = label_map[v] if v < len(label_map) else str(v)
        return lab, 1.0, int(v)
    except Exception:
        return ("unknown", 0.0, str(pred))

def compute_real_next_time_now(interval):
    """Return next interval timestamp based on Asia/Manila current clock (tz-aware)."""
    try:
        tz = ZoneInfo("Asia/Manila")
    except Exception:
        tz = ZoneInfo("Asia/Manila")
    now = datetime.now(tz)
    if interval == "60m":
        next_hour = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
        return pd.to_datetime(next_hour).tz_localize(tz) if getattr(pd.to_datetime(next_hour), "tzinfo", None) is None else pd.to_datetime(next_hour).tz_convert(tz)
    if interval == "1d":
        next_day = (now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
        return pd.to_datetime(next_day).tz_localize(tz) if getattr(pd.to_datetime(next_day), "tzinfo", None) is None else pd.to_datetime(next_day).tz_convert(tz)
    if isinstance(interval, str) and interval.endswith("m"):
        try:
            mins = int(interval[:-1])
            epoch = now
            minutes = epoch.minute
            extra = mins - (minutes % mins) if (minutes % mins) != 0 else 0
            rounded = (epoch.replace(second=0, microsecond=0) + timedelta(minutes=extra))
            return pd.to_datetime(rounded).tz_localize(tz)
        except Exception:
            pass
    return pd.to_datetime(now).tz_localize(tz)

def ensure_timestamp_in_manila(ts):
    """
    Take ts (pandas Timestamp or convertible), return tz-aware Timestamp in Asia/Manila.
    If ts is tz-naive, assume UTC then convert to Manila.
    """
    try:
        tz = ZoneInfo("Asia/Manila")
    except Exception:
        tz = ZoneInfo("Asia/Manila")

    ts = pd.to_datetime(ts)
    if getattr(ts, "tzinfo", None) is None:
        try:
            ts = ts.tz_localize("UTC").tz_convert(tz)
        except Exception:
            ts = ts.tz_localize(tz)
    else:
        try:
            ts = ts.tz_convert(tz)
        except Exception:
            pass
    return ts

def map_label_to_suggestion(label):
    """Map canonical label ('up'/'down'/'neutral') to suggestion text."""
    l = (label or "").strip().lower()
    if l == "up":
        return "Buy"
    if l == "down":
        return "Sell"
    return "Hold"

# ------------------------------
# Prediction history helpers
# ------------------------------
def evaluate_history(history_df, aligned_close_df, current_fetched_ts):
    """
    Robust evaluation of pending history rows.

    - Normalizes datetimes to Asia/Manila (uses ensure_timestamp_in_manila).
    - Normalizes tickers to UPPERCASE to match aligned_close_df columns.
    - Normalizes predicted labels to canonical {'up','down','neutral'}; if ambiguous, sets eval_error and skips evaluation.
    - Finds pred_price and actual_price using a safe "price at or before timestamp" helper (safer than raw asof).
    - Marks evaluated only when both prices are available and computes pct_change and correctness using neutral_threshold.
    - Writes helpful fields: pred_price, actual_price, pct_change, correct (bool/None), checked_at, eval_error.
    """
    if history_df is None or history_df.empty:
        return history_df

    # ensure useful columns exist
    if 'evaluated' not in history_df.columns:
        history_df['evaluated'] = False
    for c in ['predicted_at','fetched_last_ts','target_time','checked_at']:
        if c in history_df.columns:
            history_df[c] = pd.to_datetime(history_df[c], errors='coerce')

    # copy + normalize aligned_close
    ac = aligned_close_df.copy()
    try:
        ac.index = pd.to_datetime(ac.index)
    except Exception:
        pass
    try:
        # attempt to put price index into Manila tz (if naive assume UTC)
        if getattr(ac.index, "tz", None) is None:
            ac.index = ac.index.tz_localize("UTC").tz_convert(ZoneInfo("Asia/Manila"))
        else:
            ac.index = ac.index.tz_convert(ZoneInfo("Asia/Manila"))
    except Exception:
        pass
    ac = ac.sort_index()
    ac.columns = [str(c).upper() for c in ac.columns]

    # normalize current fetched ts
    try:
        now_ts = ensure_timestamp_in_manila(current_fetched_ts)
    except Exception:
        now_ts = ensure_timestamp_in_manila(pd.Timestamp.now())

    # helper: robust price at or before timestamp
    def price_at_or_before(series, when_ts):
        if series is None or series.empty:
            return float("nan")
        try:
            s = series.copy()
            s.index = pd.to_datetime(s.index)
            if getattr(s.index, "tz", None) is None:
                s.index = s.index.tz_localize("UTC").tz_convert(ZoneInfo("Asia/Manila"))
            else:
                s.index = s.index.tz_convert(ZoneInfo("Asia/Manila"))
        except Exception:
            s = series.copy()

        # ensure monotonic
        if not s.index.is_monotonic_increasing:
            s = s.sort_index()

        when_ts = ensure_timestamp_in_manila(when_ts)
        # if target earlier than first index -> return first valid value
        if when_ts < s.index[0]:
            first_valid = s.dropna()
            return float(first_valid.iloc[0]) if not first_valid.empty else float("nan")

        sel = s[:when_ts]
        if sel.empty:
            return float("nan")
        v = sel.iloc[-1]
        return float(v) if pd.notna(v) else float("nan")

    # canonicalize predicted_label strings to 'up'/'down'/'neutral' or None
    def canonical_label(raw):
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return None
        s = str(raw).strip().lower()
        # known phrases mapping
        if s in ('up','buy','long','bull','increase','rise'):
            return 'up'
        if s in ('down','sell','short','bear','decrease','fall','drop'):
            return 'down'
        if s in ('neutral','hold','no change','flat','unchanged','no reliable consensus','none','n/a','unknown'):
            return 'neutral'
        # sometimes stored like "Up" or "UP" or "BUY"
        if 'up' in s and 'down' not in s:
            return 'up'
        if 'down' in s and 'up' not in s:
            return 'down'
        if 'hold' in s or 'neutral' in s or 'no reliable' in s:
            return 'neutral'
        return None

    # prepare normalized target_time column
    history_df['_target_norm'] = history_df['target_time'].apply(lambda x: (ensure_timestamp_in_manila(x) if pd.notna(x) else pd.NaT))

    # rows ready for evaluation: not evaluated and target_time <= now_ts
    mask_ready = (~history_df['evaluated'].astype(bool)) & history_df['_target_norm'].notna() & (history_df['_target_norm'] <= now_ts)

    for idx in history_df[mask_ready].index:
        try:
            row = history_df.loc[idx]
            t_target = history_df.at[idx, '_target_norm']
            tk_raw = row.get('ticker', '')
            tk = str(tk_raw).upper().strip() if pd.notna(tk_raw) else ''
            # always stamp checked_at
            history_df.at[idx, 'checked_at'] = pd.Timestamp.now(tz=ZoneInfo("Asia/Manila"))

            if not tk:
                history_df.at[idx, 'eval_error'] = "missing_ticker"
                continue
            if tk not in ac.columns:
                history_df.at[idx, 'eval_error'] = f"ticker_not_in_price_matrix:{tk}"
                continue

            series = ac[tk].dropna()
            if series.empty:
                history_df.at[idx, 'eval_error'] = "no_price_data_for_ticker"
                continue

            # canonical predicted label
            raw_label = row.get('predicted_label', None)
            pred_label = canonical_label(raw_label)
            if pred_label is None:
                history_df.at[idx, 'eval_error'] = f"ambiguous_predicted_label:{raw_label}"
                continue

            # determine pred_price: prefer stored pred_price; else use fetched_last_ts
            pred_price_val = None
            if pd.notna(row.get('pred_price')):
                try:
                    pred_price_val = float(row.get('pred_price'))
                except Exception:
                    pred_price_val = None

            if pred_price_val is None:
                fetched_ts = row.get('fetched_last_ts')
                if pd.notna(fetched_ts):
                    pred_price_val = price_at_or_before(series, fetched_ts)
                else:
                    # fallback: price right before target
                    pred_price_val = price_at_or_before(series, t_target - pd.Timedelta(seconds=1))

            actual_price_val = price_at_or_before(series, t_target)

            # save intermediate found prices (could be nan)
            history_df.at[idx, 'pred_price'] = (None if pd.isna(pred_price_val) else float(pred_price_val))
            history_df.at[idx, 'actual_price'] = (None if pd.isna(actual_price_val) else float(actual_price_val))

            if pd.isna(pred_price_val) or pd.isna(actual_price_val):
                history_df.at[idx, 'eval_error'] = "missing_pred_or_actual_price"
                history_df.at[idx, 'pct_change'] = None
                history_df.at[idx, 'correct'] = None
                continue

            if pred_price_val == 0:
                history_df.at[idx, 'eval_error'] = "pred_price_zero"
                history_df.at[idx, 'pct_change'] = None
                history_df.at[idx, 'correct'] = None
                continue

            pct = (float(actual_price_val) - float(pred_price_val)) / float(pred_price_val)
            history_df.at[idx, 'pct_change'] = pct

            thr = float(neutral_threshold) if 'neutral_threshold' in globals() else 0.002

            # decision rules:
            is_correct = False
            if pred_label == 'up':
                is_correct = (pct > 0)
            elif pred_label == 'down':
                is_correct = (pct < 0)
            elif pred_label == 'neutral':
                is_correct = (abs(pct) <= thr)

            history_df.at[idx, 'correct'] = bool(is_correct)
            history_df.at[idx, 'evaluated'] = True
            history_df.at[idx, 'eval_error'] = None

        except Exception as e:
            # record error but continue
            try:
                history_df.at[idx, 'eval_error'] = f"exception:{str(e)}"
                history_df.at[idx, 'checked_at'] = pd.Timestamp.now(tz=ZoneInfo("Asia/Manila"))
            except Exception:
                pass
            continue

    # cleanup helper column
    if '_target_norm' in history_df.columns:
        history_df = history_df.drop(columns=['_target_norm'])

    return history_df

# ------------------------------
# Tabs
# ------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Live Market View", "Predictions", "Detailed Analysis", "History Predictions"])

# ------------------------------
# Run pipeline
# ------------------------------
if run_button:
    # 1) download data (transient spinner) for modeling & history eval
    with st.spinner("Downloading latest market data (transient)..."):
        needed = {ticker, "SPY", "QQQ", "NVDA"}
        raw = {}
        for t in needed:
            try:
                df = yf.download(t, interval=interval, period=period, progress=False)
                if df is None or df.empty:
                    continue
                df = df.dropna(how="all")
                raw[t] = df
            except Exception as e:
                st.warning(f"Download failed for {t}: {e}")

    if ticker not in raw:
        st.error(f"No data for {ticker}. Try increasing period or check connectivity.")
        st.stop()

    # Align close prices using base index; keep original tz info
    base_idx = raw[ticker].index
    aligned_close = pd.DataFrame(index=base_idx)
    for t, df in raw.items():
        aligned_close[t] = df.reindex(base_idx)["Close"]

    # fetched latest timestamp (may be tz-aware or naive)
    fetched_last_ts = base_idx[-1]
    fetched_last_ts_manila = ensure_timestamp_in_manila(fetched_last_ts)

    # compute real next interval from clock
    real_next_time = compute_real_next_time_now(interval)

    # compute now in Manila
    try:
        tz_manila = ZoneInfo("Asia/Manila")
    except Exception:
        tz_manila = ZoneInfo("Asia/Manila")
    now_manila = datetime.now(tz_manila)

    # compute age safely (both tz-aware)
    try:
        age_secs = (now_manila - fetched_last_ts_manila).total_seconds()
    except Exception:
        age_secs = (now_manila.replace(tzinfo=None) - pd.to_datetime(fetched_last_ts_manila).replace(tzinfo=None)).total_seconds()

    # determine staleness: allow tolerance of 1.5 intervals
    interval_seconds = 3600 if interval == "60m" else 86400
    tolerance = interval_seconds * 1.5
    data_is_stale = age_secs > tolerance

    # --- load and evaluate past history (if any) ---
    try:
        history = load_history()
        history = evaluate_history(history, aligned_close, fetched_last_ts_manila)
        save_history(history)
    except Exception as e:
        st.warning(f"History evaluation failed: {e}")

    # Build features for chosen ticker
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

    # ------------------------------
    # Live Market View tab (updated: single-line intraday "trading" chart)
    # ------------------------------
    with tab1:
        st.subheader(f"Live Market View — {ticker}")
        st.write(f"Fetched latest timestamp (converted to Manila): **{fetched_last_ts_manila}**")
        st.write(f"Real-time clock now (Asia/Manila): **{now_manila.strftime('%Y-%m-%d %H:%M:%S %Z')}**")
        st.write(f"Real next-interval target (clock-based): **{real_next_time}**")
        if data_is_stale:
            st.caption("Note: fetched market data is stale relative to the real-time clock and may lag real-time.")

        # We attempt to fetch the highest-resolution intraday available (yfinance supports 1m)
        intraday_df = pd.DataFrame()
        try:
            tk = yf.Ticker(ticker)
            with st.spinner("Fetching intraday ticks (1m granularity)..."):
                intraday_df = tk.history(period="1d", interval="1m", prepost=False, actions=False, auto_adjust=False)
            if intraday_df is None or intraday_df.empty:
                # fallback: use the raw data we downloaded earlier (if present)
                intraday_df = raw.get(ticker, pd.DataFrame()).copy()
        except Exception:
            intraday_df = raw.get(ticker, pd.DataFrame()).copy()

        chart_df = intraday_df.copy()  # expose to Detailed Analysis later
        if chart_df is None or chart_df.empty:
            st.info("Intraday tick data not available to draw the realtime line chart.")
        else:
            # ensure datetime index
            try:
                chart_df.index = pd.to_datetime(chart_df.index)
            except Exception:
                pass

            # get price series
            price_series = chart_df['Close'].ffill().dropna()
            if price_series.empty:
                st.info("No intraday close prices available.")
            else:
                # compute previous close/reference: try 2-day daily history
                prev_close = None
                try:
                    hist2d = tk.history(period="2d", interval="1d", auto_adjust=False)
                    if hist2d is not None and len(hist2d) >= 2:
                        # previous day's close is the penultimate row
                        prev_close = float(hist2d['Close'].iloc[-2])
                    else:
                        # fallback: first recorded price in intraday
                        prev_close = float(price_series.iloc[0])
                except Exception:
                    try:
                        prev_close = float(price_series.iloc[0])
                    except Exception:
                        prev_close = None

                current_price = float(price_series.iloc[-1])
                current_time = price_series.index[-1]

                # build dark-themed single-line chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=price_series.index,
                    y=price_series.values,
                    mode='lines',
                    name=f"{ticker} Price",
                    line=dict(width=2, color="#00E5FF")  # cyan-ish line
                ))

                # current price horizontal dashed red line + marker
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

                # previous close as white dashed line
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

                # last-price marker
                fig.add_trace(go.Scatter(
                    x=[current_time],
                    y=[current_price],
                    mode='markers',
                    marker=dict(color='red', size=8),
                    name='Current Price'
                ))

                # dark theme styling (TradingView-like)
                fig.update_layout(
                    plot_bgcolor="#0b1020",   # dark plot bg
                    paper_bgcolor="#0b1020",
                    font_color="white",
                    height=520,
                    hovermode="x unified",
                    margin=dict(l=10, r=10, t=30, b=30),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                # dotted / subtle grid
                fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", gridwidth=1, zeroline=False, color="white")
                fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", gridwidth=1, zeroline=False, color="white")

                # add range selector for UX (1h, 6h, 1d, all)
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

    # Load models
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
            mod = safe_load(path)
            if mod is not None:
                loaded[name] = mod

    if not loaded:
        st.error("No models loaded. Place .pkl files next to app.py or correct paths.")
        st.stop()

    # prepare last_row and seq_df
    last_row = feat_df.drop(columns=["datetime","ticker"], errors='ignore').iloc[-1:].replace([np.inf,-np.inf], np.nan).fillna(0.0)
    seq_df = feat_df.drop(columns=["datetime","ticker"], errors='ignore').replace([np.inf,-np.inf], np.nan).fillna(0.0)

    # predict per-model
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
                    raw_pred = mod.predict(X_in.reshape((1,-1)))
                lab, conf, rawinfo = label_from_model(mod, X_in)

                # normalize label to canonical up/down/neutral when possible
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

                        # normalize label
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

    # Display Predictions tab
    with tab2:
        st.subheader("Predictions — next interval (real-time)")
        st.write(f"Ticker: **{ticker}**   •   Fetched latest (Manila): **{fetched_last_ts_manila}**")
        st.write(f"Real-time clock now (Asia/Manila): **{now_manila.strftime('%Y-%m-%d %H:%M:%S %Z')}**")
        st.write(f"Real next-interval target (clock-based): **{real_next_time}**")
        if data_is_stale:
            st.caption("Note: fetched market data is stale relative to the real-time clock and may lag real-time.")

        # show predictions table (no colored suggestions)
        if results:
            out = pd.DataFrame(results)[["model", "label", "confidence", "suggestion", "raw"]]
            try:
                st.dataframe(out)
            except Exception:
                st.dataframe(out)
        else:
            st.warning("No predictions produced.")

        # Conclusive meta (preferred) with ensemble fallback
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

        # show conclusive box (label is Up/Down/Neutral)
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

        # before download: append this prediction to history and show history for ticker
        try:
            # load existing history robustly
            history = load_history()
        except Exception:
            history = pd.DataFrame()

        try:
            pred_price = None
            if ticker in aligned_close.columns:
                try:
                    pred_price = float(aligned_close[ticker].asof(fetched_last_ts_manila))
                except Exception:
                    try:
                        pred_price = float(aligned_close[ticker].loc[fetched_last_ts_manila])
                    except Exception:
                        pred_price = None
        except Exception:
            pred_price = None

        new_row = {
            'predicted_at': pd.Timestamp.now(tz_manila),
            'ticker': ticker,
            'interval': interval,
            'predicted_label': con_label if isinstance(con_label, str) else str(con_label),
            'suggestion': map_label_to_suggestion(con_label.lower()) if isinstance(con_label, str) else '',
            'confidence': float(con_conf) if con_conf is not None else None,
            'fetched_last_ts': fetched_last_ts_manila,
            'target_time': real_next_time,
            'pred_price': pred_price,
            'evaluated': False,
            'actual_price': None,
            'pct_change': None,
            'correct': None,
            'checked_at': None
        }

        # robust append: always ensure same columns and use concat
        try:
            if history is None or history.empty:
                history = pd.DataFrame([new_row])
            else:
                # ensure columns exist
                missing_cols = set(new_row.keys()) - set(history.columns)
                for mc in missing_cols:
                    history[mc] = pd.NA
                history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True, sort=False)
            save_history(history)
        except Exception as e:
            st.warning(f"Failed to append/save history: {e}")

        # show history filtered for this ticker
        try:
            history = load_history()
            if not history.empty:
                hist_t = history[history['ticker'].str.upper() == ticker.upper()].sort_values('predicted_at', ascending=False)
                if not hist_t.empty:
                    st.subheader('Prediction history — this ticker')
                    cols = ['predicted_at','target_time','predicted_label','suggestion','confidence','pred_price','actual_price','pct_change','correct','evaluated','checked_at']
                    available = [c for c in cols if c in hist_t.columns]
                    st.dataframe(hist_t[available].head(50))
        except Exception:
            pass

        # download
        if results:
            st.download_button("Download per-model predictions CSV", data=pd.DataFrame(results).to_csv(index=False).encode("utf-8"), file_name="predictions_next_interval.csv")

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
                        orientation='h',
                        marker=dict(color=['green','gray','red'])
                    ))
                    fig_prob.update_layout(height=250, xaxis_title='Probability (%)')
                    st.plotly_chart(fig_prob, use_container_width=True)
                except Exception:
                    pass

                try:
                    val = float(con_conf) * 100 if con_conf is not None else 0.0
                    fig_gauge = go.Figure(go.Indicator(mode='gauge+number', value=val,
                                                      gauge={'axis':{'range':[0,100]}, 'bar':{'color':'darkblue'}}))
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





with tab4:
    st.subheader("History Prediction Section")

    history = load_history()
    if history is None or history.empty:
        st.info("No prediction history found. Predictions will be recorded here after you run the model.")
    else:
        # normalize datetime columns
        for c in ['predicted_at','fetched_last_ts','target_time','checked_at']:
            if c in history.columns:
                history[c] = pd.to_datetime(history[c], errors='coerce')

        # Force re-evaluation button (user requested)
        if st.button("Force re-evaluate pending predictions"):
            eval_ph = st.empty()
            eval_ph.info("Force re-evaluation started — fetching market data and attempting to evaluate pending predictions...")
            unevaluated = history[~history['evaluated'].astype(bool)]
            if unevaluated.empty:
                eval_ph.info("No pending predictions to evaluate.")
                eval_ph.empty()
            else:
                raw_price_map = {}
                groups = unevaluated.groupby(['ticker','interval'])
                for (tk, iv), grp in groups:
                    try:
                        yf_interval = iv if isinstance(iv, str) else "60m"
                        min_fetched = grp['fetched_last_ts'].min()
                        max_target = grp['target_time'].max()
                        start = (pd.to_datetime(min_fetched) - pd.Timedelta(days=2)).date()
                        end = (pd.to_datetime(max_target) + pd.Timedelta(days=1)).date()
                        df = None
                        try:
                            df = yf.download(tk, start=start, end=end, interval=yf_interval, progress=False)
                        except Exception as e:
                            st.warning(f"Download failed for {tk} ({iv}): {e}")
                        if df is not None and not df.empty:
                            raw_price_map[tk.upper()] = df
                    except Exception as e:
                        st.warning(f"Failed to prepare download for {tk}: {e}")

                if raw_price_map:
                    all_index = sorted(set().union(*(df.index for df in raw_price_map.values())))
                    aligned_close_for_eval = pd.DataFrame(index=all_index)
                    for t, df in raw_price_map.items():
                        aligned_close_for_eval[t] = df.reindex(all_index)["Close"]
                    try:
                        now_manila = datetime.now(ZoneInfo("Asia/Manila"))
                    except Exception:
                        now_manila = datetime.utcnow()
                    try:
                        history = evaluate_history(history, aligned_close_for_eval, now_manila)
                        save_history(history)
                        eval_ph.empty()
                        st.write("Force evaluation attempted and history updated where possible.")
                    except Exception as e:
                        st.warning(f"Force evaluation attempt failed: {e}")
                else:
                    eval_ph.info("Could not fetch price series for the pending tickers — they will be re-attempted later.")

        # Auto-evaluate pending predictions silently (attempt on open)
        unevaluated = history[~history['evaluated'].astype(bool)]
        if not unevaluated.empty:
            eval_placeholder = st.empty()
            eval_placeholder.info(f"Found {len(unevaluated)} unevaluated prediction(s). Attempting to fetch market data to evaluate them...")
            raw_price_map = {}
            groups = unevaluated.groupby(['ticker','interval'])
            for (tk, iv), grp in groups:
                try:
                    yf_interval = iv if isinstance(iv, str) else "60m"
                    min_fetched = grp['fetched_last_ts'].min()
                    max_target = grp['target_time'].max()
                    start = (pd.to_datetime(min_fetched) - pd.Timedelta(days=2)).date()
                    end = (pd.to_datetime(max_target) + pd.Timedelta(days=1)).date()
                    df = None
                    try:
                        df = yf.download(tk, start=start, end=end, interval=yf_interval, progress=False)
                    except Exception as e:
                        st.warning(f"Download failed for {tk} ({iv}): {e}")
                    if df is not None and not df.empty:
                        raw_price_map[tk.upper()] = df
                except Exception as e:
                    st.warning(f"Failed to prepare download for {tk}: {e}")

            if raw_price_map:
                all_index = sorted(set().union(*(df.index for df in raw_price_map.values())))
                aligned_close_for_eval = pd.DataFrame(index=all_index)
                for t, df in raw_price_map.items():
                    aligned_close_for_eval[t] = df.reindex(all_index)["Close"]
                try:
                    now_manila = datetime.now(ZoneInfo("Asia/Manila"))
                except Exception:
                    now_manila = datetime.utcnow()
                try:
                    history = evaluate_history(history, aligned_close_for_eval, now_manila)
                    save_history(history)
                    eval_placeholder.empty()
                    st.write("Evaluation attempted and history updated where possible.")
                except Exception as e:
                    st.warning(f"Evaluation attempt failed: {e}")
            else:
                eval_placeholder.info("Could not fetch price series for the pending tickers — they will be re-attempted on the next run or when market data is available.")

        # Re-load after attempted evaluation
        history = load_history()

        # --- 1) Summary Accuracy Stats ---
        total_preds = len(history)
        evaluated_rows = history[history['evaluated'].astype(bool)]
        evaluated_count = len(evaluated_rows)
        correct_count = int(evaluated_rows['correct'].sum()) if evaluated_count > 0 else 0
        accuracy = (correct_count / evaluated_count) if evaluated_count > 0 else 0.0

        st.markdown("### Summary Accuracy Stats")
        cols = st.columns([2, 1])
        with cols[0]:
            st.write(f"**Total Predictions Made:** {total_preds}")
            st.write(f"**Correct Predictions:** {correct_count}")
            st.write(f"**Evaluated:** {evaluated_count}")
            st.write(f"**Accuracy:** {accuracy:.1%}" if evaluated_count > 0 else "**Accuracy:** N/A (no evaluated rows)")

        with cols[1]:
            # donut chart: correct / incorrect / pending
            pending_count = total_preds - evaluated_count
            incorrect = evaluated_count - correct_count
            pie_vals = [correct_count, incorrect, pending_count]
            pie_labels = ["Correct", "Incorrect", "Pending"]
            try:
                fig_donut = go.Figure(go.Pie(labels=pie_labels, values=pie_vals, hole=0.6,
                                            marker=dict(colors=["#16a34a","#ef4444","#9ca3af"])))
                fig_donut.update_layout(showlegend=True, margin=dict(t=0,b=0,l=0,r=0), height=200)
                st.plotly_chart(fig_donut, use_container_width=True)
            except Exception:
                st.write(f"Correct: {correct_count}  Incorrect: {incorrect}  Pending: {pending_count}")

        # --- 2) Recent Predictions Table ---
        st.markdown("### Recent Predictions Table")
        display_rows = []
        for _, row in history.sort_values('predicted_at', ascending=False).head(200).iterrows():
            pred_at = row.get('predicted_at')
            tk = str(row.get('ticker','')).upper()
            pred_lbl = str(row.get('predicted_label','')).upper() if pd.notna(row.get('predicted_label')) else ""
            actual_move = ""
            correct_mark = "⏳"
            if pd.notna(row.get('actual_price')) and pd.notna(row.get('pred_price')):
                try:
                    pct = float(row.get('pct_change'))
                    thr = float(neutral_threshold)
                    if abs(pct) <= thr:
                        actual_move = "NEUTRAL"
                    elif pct > 0:
                        actual_move = "UP"
                    else:
                        actual_move = "DOWN"
                except Exception:
                    actual_move = ""
            else:
                if pd.notna(row.get('target_time')):
                    try:
                        if pd.to_datetime(row.get('target_time')) <= pd.Timestamp.now(tz=ZoneInfo("Asia/Manila")):
                            actual_move = ""
                        else:
                            actual_move = ""
                    except Exception:
                        actual_move = ""
            if pd.notna(row.get('correct')):
                correct_mark = "✅" if bool(row.get('correct')) else "❌"
            else:
                correct_mark = "⏳"

            display_rows.append({
                "Date/Time": pred_at,
                "Ticker": tk,
                "Predicted Movement": pred_lbl,
                "Actual Movement": actual_move,
                "Correct?": correct_mark
            })

        df_display = pd.DataFrame(display_rows)
        if not df_display.empty:
            try:
                st.dataframe(df_display)
            except Exception:
                st.write(df_display)
        else:
            st.write("No rows to show.")

        # allow download of the filtered history CSV
        try:
            csv_bytes = history.to_csv(index=False).encode("utf-8")
            st.download_button("Download full history CSV", data=csv_bytes, file_name="predictions_history.csv")
        except Exception:
            pass
