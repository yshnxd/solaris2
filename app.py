# SOLARIS — Stock Price Movement Predictor (adapted for the notebook pipeline)
# - Updated to match the notebook's feature/sequence shape and meta-learner usage
# - Loads CNN/LSTM (keras or joblib), XGB (sklearn/xgboost) and a regression meta-learner
# - Builds features close to the notebook (lag returns, rolling vol, RSI, MACD, SMA/EMA)
# - Scales features with a saved scaler (scaler.pkl) if provided
# - Creates sequences for CNN/LSTM and aligns tabular rows for XGB/meta
# - Produces regression prediction (next-hour return) and maps to Buy/Hold/Sell using user threshold
# - Persists predictions to a CSV history with dedupe (by ticker/interval/target_time)
import os
import tempfile
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

# -------------------------------------------------------------------
# Persisted history utilities (session_state-backed, atomic writes)
# -------------------------------------------------------------------
def get_history_file():
    return st.session_state.get('history_file', st.sidebar.text_input("History CSV file", "predictions_history.csv"))

history_file = get_history_file()

def load_history():
    if "history" not in st.session_state:
        if os.path.exists(history_file):
            df = pd.read_csv(history_file)
            for c in ["predicted_at","fetched_last_ts","target_time","checked_at"]:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors='coerce')
            st.session_state["history"] = df
        else:
            st.session_state["history"] = pd.DataFrame()
    return st.session_state["history"]

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
        st.warning(f"Failed to save history atomically to {target_path}: {e}")
        try:
            df.to_csv(target_path, index=False)
        except Exception as e2:
            st.warning(f"Fallback save also failed: {e2}")
    st.session_state["history"] = df

def append_prediction_with_dedup(history, new_row):
    import pandas as pd
    if history is None or (hasattr(history,'empty') and history.empty):
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
        # Deduplicate: keep latest prediction per (ticker, interval, target_time)
        dedup_cols = ['ticker','interval','target_time']
        if 'predicted_at' in out.columns:
            out = out.sort_values('predicted_at', ascending=False).drop_duplicates(subset=dedup_cols, keep='first')
        else:
            out = out.drop_duplicates(subset=dedup_cols, keep='last')
    return out

# -------------------------------------------------------------------
# Page & Sidebar UI
# -------------------------------------------------------------------
st.set_page_config(page_title="Solaris - Predict Next Hour!", layout="wide")
st.title("SOLARIS — Stock Price Movement Predictor")
st.markdown("_Adapted to your notebook pipeline: regression meta-learner + CNN/LSTM/XGB ensembler_")

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker (Yahoo symbol)", "AAPL").strip().upper()

sequence_length = st.sidebar.number_input("Sequence length (for CNN/LSTM)", min_value=3, max_value=512, value=128, step=1)
interval = st.sidebar.selectbox("Interval", ["60m"], index=0)
period_default = "90d" if interval == "60m" else "90d"
period = st.sidebar.text_input("Period (try 90d, 365d, or 729d)", period_default)

st.sidebar.markdown("---")
st.sidebar.write("Model files (relative to app.py):")
cnn_path   = st.sidebar.text_input("CNN (.h5/.pkl)", "cnn_model.pkl")
lstm_path  = st.sidebar.text_input("LSTM (.h5/.pkl)", "lstm_model.pkl")
xgb_path   = st.sidebar.text_input("XGB (.pkl)", "xgb_model.pkl")
meta_path  = st.sidebar.text_input("Meta (.pkl)", "meta_learner.pkl")
scaler_path = st.sidebar.text_input("Scaler (.pkl) (StandardScaler)", "scaler.pkl")

st.sidebar.markdown("---")
st.sidebar.write("Label mapping / thresholds")
label_map_input = st.sidebar.text_input("Labels (index order)", "down,neutral,up")
label_map = [s.strip().lower() for s in label_map_input.split(",")]
# Threshold for mapping a regression return to buy/sell/hold (absolute return)
return_threshold = st.sidebar.number_input("Return threshold (absolute, e.g. 0.001 = 0.1%)", value=0.001, format="%.6f")
prob_threshold = st.sidebar.slider("Conf threshold (unused for regression mapping)", 0.05, 0.95, 0.5, 0.05)

models_to_run = st.sidebar.multiselect("Models to run", ["cnn", "lstm", "xgb", "meta"], default=["cnn", "lstm", "xgb", "meta"])
run_button = st.sidebar.button("Fetch live data & predict NEXT interval (real-time)")

COLOR_BUY_BG = "#d7e9da"   # soft green
COLOR_SELL_BG = "#f0e5e6"  # soft rose
COLOR_HOLD_BG = "#f3f4f6"  # soft gray
COLOR_TEXT = "#111111"

# -------------------------------------------------------------------
# Helpers: model loading / caching / pricing / time handling / features
# -------------------------------------------------------------------
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

# Feature engineering similar to the notebook (single-ticker simplified version)
def create_features_from_price_df(df: pd.DataFrame):
    # df expected to have columns: Open High Low Close Volume
    X = pd.DataFrame(index=df.index)
    price = df['Close']
    X['ret_1h'] = price.pct_change(1)
    X['ret_3h'] = price.pct_change(3)
    X['ret_6h'] = price.pct_change(6)
    X['ret_12h'] = price.pct_change(12)
    X['ret_24h'] = price.pct_change(24)
    for w in [6, 12, 24]:
        X[f'vol_{w}h'] = price.pct_change().rolling(w).std()
    # RSI 14
    try:
        X['rsi_14'] = ta.momentum.RSIIndicator(price, window=14).rsi()
    except Exception:
        X['rsi_14'] = np.nan
    # MACD
    try:
        macd = ta.trend.MACD(price)
        X['macd'] = macd.macd()
        X['macd_signal'] = macd.macd_signal()
    except Exception:
        X['macd'] = np.nan
        X['macd_signal'] = np.nan
    # Moving averages
    for w in [5, 10, 20]:
        X[f'sma_{w}'] = price.rolling(w).mean()
        X[f'ema_{w}'] = price.ewm(span=w, adjust=False).mean()
    # Volume based
    if 'Volume' in df.columns:
        vol = df['Volume'].ffill()
        X['vol_change_1h'] = vol.pct_change()
        X['vol_ma_24h'] = vol.rolling(24).mean()
        X['vol'] = vol
    # calendar
    X['hour'] = X.index.hour
    X['day_of_week'] = X.index.dayofweek
    return X

def create_sequences(X_arr: np.ndarray, seq_len=128):
    X_seq = []
    for i in range(len(X_arr) - seq_len + 1):
        X_seq.append(X_arr[i:i+seq_len])
    return np.array(X_seq)

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

def map_return_to_label(ret: float, threshold: float):
    if ret is None or np.isnan(ret):
        return "neutral"
    if ret > threshold:
        return "up"
    if ret < -threshold:
        return "down"
    return "neutral"

def map_label_to_suggestion(label):
    l = (label or "").strip().lower()
    if l == "up":
        return "Buy"
    if l == "down":
        return "Sell"
    return "Hold"

# -------------------------------------------------------------------
# Load models / scaler (cached)
# -------------------------------------------------------------------
cnn_model = load_model_safe(cnn_path) if "cnn" in models_to_run else None
lstm_model = load_model_safe(lstm_path) if "lstm" in models_to_run else None
xgb_model = load_model_safe(xgb_path) if "xgb" in models_to_run else None
meta_model = load_model_safe(meta_path) if "meta" in models_to_run else None

scaler = None
if scaler_path and os.path.exists(scaler_path):
    try:
        scaler = joblib.load(scaler_path)
    except Exception:
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            warn(f"Failed to load scaler: {e}")
            scaler = None

# -------------------------------------------------------------------
# Main action: fetch data, build features, make predictions, persist
# -------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.write("History & persistence")
if st.sidebar.button("Reload history"):
    st.session_state.pop("history", None)
history_df = load_history()

if run_button:
    with st.spinner("Downloading data and preparing features..."):
        price_df = download_price_cached(ticker, interval, period)
        if price_df.empty:
            st.error("No price data downloaded. Check ticker/interval/period or network.")
        else:
            st.success(f"Downloaded {len(price_df)} rows for {ticker} ({interval})")

            # Create features
            feat_df = create_features_from_price_df(price_df)
            feat_df = feat_df.dropna()
            if feat_df.empty:
                st.error("No features (all NaN) after feature engineering.")
            else:
                # Determine required feature names from models if possible
                # Prefer meta learner alpha: expects 7 features [cnn_pred, lstm_pred, xgb_pred, mean,std,max,min]
                # For base models, determine expected tabular features for XGB (if possible)
                xgb_expected = None
                if xgb_model is not None:
                    # try to infer n features or feature names
                    fn = getattr(xgb_model, "feature_names_in_", None)
                    if fn is not None:
                        xgb_expected = list(fn)
                # If scaler exists, we will scale the tabular X columns that were used in training.
                # We'll try to align scaler.feature_names_in_ if available
                scaler_expected = None
                if scaler is not None:
                    scaler_expected = getattr(scaler, "feature_names_in_", None)
                    if scaler_expected is not None:
                        scaler_expected = list(scaler_expected)
                # Build tabular matrix (last N rows) and sequence matrix (for seq_len)
                seq_len = int(sequence_length)
                # Select last rows enough for sequence
                if len(feat_df) < seq_len:
                    st.error(f"Not enough rows to create one sequence (need {seq_len}, got {len(feat_df)})")
                else:
                    last_feat_df = feat_df.copy().iloc[-(seq_len+1):]  # +1 to align label/time
                    # align features order to scaler or xgb if provided; otherwise use current columns
                    base_feature_names = list(feat_df.columns)
                    if scaler_expected:
                        # take only names present in feat_df & scaler_expected (preserve scaler order)
                        common = [n for n in scaler_expected if n in feat_df.columns]
                        if len(common) >= 1:
                            base_feature_names = common
                    elif xgb_expected:
                        common = [n for n in xgb_expected if n in feat_df.columns]
                        if len(common) >= 1:
                            base_feature_names = common

                    tab_X_last = last_feat_df[base_feature_names].iloc[-1:].fillna(0.0)
                    # scale tabular
                    tab_X_scaled = tab_X_last.copy()
                    if scaler is not None:
                        try:
                            # If scaler was fitted to a full array (no feature_names), call transform directly
                            if scaler_expected is None:
                                arr = last_feat_df[base_feature_names].iloc[-1:].to_numpy().astype(float)
                                arr_s = scaler.transform(arr)
                                tab_X_scaled = pd.DataFrame(arr_s, columns=base_feature_names)
                            else:
                                # Ensure ordering
                                arr = last_feat_df[scaler_expected].iloc[-1:].to_numpy().astype(float)
                                arr_s = scaler.transform(arr)
                                tab_X_scaled = pd.DataFrame(arr_s, columns=scaler_expected)
                                base_feature_names = scaler_expected
                        except Exception as e:
                            warn(f"Scaler transform failed: {e}")
                            tab_X_scaled = tab_X_last.fillna(0.0)

                    # Prepare sequence array for CNN/LSTM: need seq_len x n_features
                    seq_df_for_model = last_feat_df.iloc[-seq_len:][base_feature_names].fillna(0.0)
                    X_seq_arr = seq_df_for_model.to_numpy(dtype=float)
                    # if sequence model expects scaled features, use same scaler transform
                    if scaler is not None:
                        try:
                            if scaler_expected is None:
                                X_seq_arr = scaler.transform(seq_df_for_model.values)
                            else:
                                # reorder features for scaler_expected if necessary
                                seq_df_for_model2 = seq_df_for_model.copy()
                                missing = [c for c in scaler_expected if c not in seq_df_for_model2.columns]
                                for m in missing:
                                    seq_df_for_model2[m] = 0.0
                                seq_df_for_model2 = seq_df_for_model2[scaler_expected]
                                X_seq_arr = scaler.transform(seq_df_for_model2.values)
                                base_feature_names = scaler_expected
                        except Exception as e:
                            warn(f"Scaler transform for sequence failed: {e}")

                    # reshape for model predict: (1, seq_len, n_features)
                    X_seq_input = X_seq_arr.reshape(1, X_seq_arr.shape[0], X_seq_arr.shape[1])

                    # Predictions: CNN/LSTM/XGB (regression outputs)
                    preds = {}
                    # CNN
                    if cnn_model is not None and "cnn" in models_to_run:
                        try:
                            p = cnn_model.predict(X_seq_input, verbose=0)
                            # handle shape (1,1) or (1,) etc.
                            pval = float(np.asarray(p).reshape(-1)[0])
                        except Exception as e:
                            warn(f"CNN predict failed: {e}")
                            pval = np.nan
                        preds['cnn'] = pval
                    # LSTM
                    if lstm_model is not None and "lstm" in models_to_run:
                        try:
                            p = lstm_model.predict(X_seq_input, verbose=0)
                            pval = float(np.asarray(p).reshape(-1)[0])
                        except Exception as e:
                            warn(f"LSTM predict failed: {e}")
                            pval = np.nan
                        preds['lstm'] = pval
                    # XGB
                    if xgb_model is not None and "xgb" in models_to_run:
                        try:
                            # xgb expects 2D array with same columns used in training
                            arr_tab = tab_X_scaled.to_numpy(dtype=float)
                            p = xgb_model.predict(arr_tab)
                            pval = float(np.asarray(p).reshape(-1)[0])
                        except Exception as e:
                            warn(f"XGB predict failed: {e}")
                            pval = np.nan
                        preds['xgb'] = pval

                    # Meta
                    if meta_model is not None and "meta" in models_to_run:
                        try:
                            # Build meta features: [cnn_pred, lstm_pred, xgb_pred, mean,std,max,min]
                            base_preds = []
                            for k in ['cnn','lstm','xgb']:
                                base_preds.append(preds.get(k, np.nan))
                            arr = np.array(base_preds, dtype=float)
                            mean_pred = np.nanmean(arr)
                            std_pred = np.nanstd(arr)
                            max_pred = np.nanmax(arr)
                            min_pred = np.nanmin(arr)
                            meta_X = np.array([[arr[0], arr[1], arr[2], mean_pred, std_pred, max_pred, min_pred]])
                            pmeta = meta_model.predict(meta_X)
                            pmeta_val = float(np.asarray(pmeta).reshape(-1)[0])
                        except Exception as e:
                            warn(f"Meta predict failed: {e}")
                            pmeta_val = np.nan
                        preds['meta'] = pmeta_val

                    # Build display
                    st.subheader(f"Predictions for {ticker} (next {interval})")
                    now_ts = datetime.utcnow().isoformat() + "Z"
                    next_ts = compute_real_next_time_now(interval)
                    st.write("Target prediction time (Asia/Manila):", next_ts)

                    # Compose result row
                    result = {
                        "ticker": ticker,
                        "interval": interval,
                        "predicted_at": pd.Timestamp.now(tz=ZoneInfo("Asia/Manila")),
                        "fetched_last_ts": price_df.index[-1] if len(price_df)>0 else pd.NaT,
                        "target_time": next_ts,
                        "price": float(price_df['Close'].iloc[-1]) if len(price_df)>0 else np.nan,
                        "pred_cnn": preds.get('cnn', np.nan),
                        "pred_lstm": preds.get('lstm', np.nan),
                        "pred_xgb": preds.get('xgb', np.nan),
                        "pred_meta": preds.get('meta', np.nan)
                    }

                    # Map pred_meta to label and suggestion
                    meta_return = result.get('pred_meta', np.nan)
                    label = map_return_to_label(meta_return, return_threshold) if not np.isnan(meta_return) else "neutral"
                    suggestion = map_label_to_suggestion(label)
                    result['label_meta'] = label
                    result['suggestion_meta'] = suggestion

                    # Display summary
                    cols = st.columns(4)
                    cols[0].metric("Last Close", f"{result['price']:.2f}")
                    cols[1].metric("CNN (ret)", f"{result['pred_cnn']:.6f}")
                    cols[2].metric("LSTM (ret)", f"{result['pred_lstm']:.6f}")
                    cols[3].metric("XGB (ret)", f"{result['pred_xgb']:.6f}")
                    st.info(f"Meta prediction (regression return): {result['pred_meta']:.6f}  -> Label: {label.upper()}  Suggestion: {suggestion}")

                    # Append to history and save
                    hist = load_history()
                    new_hist = append_prediction_with_dedup(hist, result)
                    try:
                        save_history_atomic(new_hist, history_file)
                        st.success(f"Saved prediction to history ({history_file})")
                    except Exception as e:
                        st.warning(f"Could not save history file: {e}")

                    # Show last N history for this ticker
                    st.subheader("Recent predictions (history)")
                    display_hist = new_hist.copy()
                    if 'predicted_at' in display_hist.columns:
                        display_hist = display_hist.sort_values('predicted_at', ascending=False)
                    st.dataframe(display_hist.head(25))

# -------------------------------------------------------------------
# Show available models & basic diagnostics
# -------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.write("Loaded models (status)")
def model_status_str(m):
    if m is None:
        return "NOT LOADED"
    try:
        # attempt to print type and shape info
        name = type(m).__name__
        nfeat = getattr(m, "n_features_in_", None)
        if nfeat is not None:
            return f"{name} (n_features={nfeat})"
        if hasattr(m, "input_shape"):
            return f"{name} (input_shape={m.input_shape})"
        return f"{name}"
    except Exception:
        return str(type(m))

st.sidebar.write("CNN: " + model_status_str(cnn_model))
st.sidebar.write("LSTM: " + model_status_str(lstm_model))
st.sidebar.write("XGB: " + model_status_str(xgb_model))
st.sidebar.write("Meta: " + model_status_str(meta_model))
st.sidebar.write("Scaler: " + (str(type(scaler).__name__) if scaler is not None else "NOT LOADED"))

# -------------------------------------------------------------------
# Show full history viewer + export
# -------------------------------------------------------------------
st.markdown("## Prediction History")
hist = load_history()
if hist is None or hist.empty:
    st.write("No history yet. Run a prediction to populate history.")
else:
    st.dataframe(hist.sort_values("predicted_at", ascending=False).head(200))
    if st.button("Download history CSV"):
        st.download_button("Download CSV", data=hist.to_csv(index=False), file_name=os.path.basename(history_file))

st.markdown("## Notes")
st.markdown("""
- This app was adapted to match your notebook pipeline (regression targets: next-hour returns).
- The meta-learner produces a continuous return prediction. Mapping to Buy/Hold/Sell uses the *Return threshold*
  you set in the sidebar (absolute return). For example, 0.001 = 0.1% next-hour move.
- The app attempts to load a scaler (StandardScaler) so that sequence/tabular inputs are transformed the same way
  as during training. If you used different preprocessing during training, ensure you provide the same scaler file.
- If your CNN/LSTM were saved as Keras .h5 files, the loader will try keras.load_model; if they were pickled by
  joblib it's also supported.
- The feature builder is a simplified single-ticker adaptation of the notebook's feature set (lags, vol, RSI, MACD, SMA/EMA).
- For production, consider stronger alignment (exact feature order used during training), and robust timezone handling.
""")
