# app.py - Solaris Reborn (Real-time Next-Hour Prediction)
import streamlit as st
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

# plotting
import plotly.graph_objects as go

# market data & indicators
import yfinance as yf
import ta

# models & preprocessing
import joblib
from sklearn.preprocessing import MinMaxScaler

# timezone / time
from datetime import datetime, timedelta
try:
    from zoneinfo import ZoneInfo     # Python 3.9+
except Exception:
    from pytz import timezone as ZoneInfo

# reproducibility
import random, tensorflow as tf
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

st.set_page_config(page_title="Solaris Reborn — Real-time Next-Hour", layout="wide")
st.title("🌞 Solaris Reborn — Real-time Next-Hour Prediction")

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker (any Yahoo Finance symbol)", "AAPL")
interval = st.sidebar.selectbox("Interval", ["60m", "1d"], index=0)
period_default = "2d" if interval == "60m" else "90d"
period = st.sidebar.text_input("Period (e.g., 2d for intraday, 90d)", period_default)
sequence_length = st.sidebar.number_input("Sequence length (LSTM/CNN)", min_value=3, max_value=240, value=24, step=1)

st.sidebar.markdown("---")
st.sidebar.write("Model files (relative to app.py):")
cnn_path = st.sidebar.text_input("CNN (.pkl)", "cnn_model.pkl")
lstm_path = st.sidebar.text_input("LSTM (.pkl)", "lstm_model.pkl")
xgb_path = st.sidebar.text_input("XGB (.pkl)", "xgb_model.pkl")
meta_path = st.sidebar.text_input("Meta (.pkl)", "meta_learner.pkl")

st.sidebar.markdown("---")
label_map_input = st.sidebar.text_input("Labels (index order)", "down,neutral,up")
label_map = [s.strip().lower() for s in label_map_input.split(",")]
prob_threshold = st.sidebar.slider("Prob threshold for confident class", 0.05, 0.95, 0.5, 0.05)
models_to_run = st.sidebar.multiselect("Models to run", ["cnn", "lstm", "xgb", "meta"], default=["cnn", "lstm", "xgb", "meta"])
run_button = st.sidebar.button("Fetch live data & predict NEXT interval (real-time)")

# muted colors (subtle)
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
        # round up to next hour
        next_hour = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
        return pd.to_datetime(next_hour).tz_localize(tz) if getattr(pd.to_datetime(next_hour), "tzinfo", None) is None else pd.to_datetime(next_hour).tz_convert(tz)
    if interval == "1d":
        next_day = (now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
        return pd.to_datetime(next_day).tz_localize(tz) if getattr(pd.to_datetime(next_day), "tzinfo", None) is None else pd.to_datetime(next_day).tz_convert(tz)
    # fallback parse minutes
    if isinstance(interval, str) and interval.endswith("m"):
        try:
            mins = int(interval[:-1])
            delta = timedelta(minutes=mins)
            # compute ceiling multiple of mins
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
        # assume UTC then convert
        try:
            ts = ts.tz_localize("UTC").tz_convert(tz)
        except Exception:
            # fallback: localize directly to Manila
            ts = ts.tz_localize(tz)
    else:
        try:
            ts = ts.tz_convert(tz)
        except Exception:
            pass
    return ts

# New helper: canonical suggestion mapping
def map_label_to_suggestion(label):
    """Map canonical label ('up'/'down'/'neutral') to suggestion text."""
    l = (label or "").strip().lower()
    if l == "up":
        return "Buy"
    if l == "down":
        return "Sell"
    return "Hold"  # neutral or anything else

# ------------------------------
# Tabs
# ------------------------------
tab1, tab2 = st.tabs(["Live Market View", "Predictions"])

# ------------------------------
# Run pipeline
# ------------------------------
if run_button:
    # 1) download data (transient spinner)
    with st.spinner("Downloading latest market data (transient)..."):
        needed = {ticker, "SPY", "QQQ", "NVDA"}
        raw = {}
        for t in needed:
            try:
                df = yf.download(t, interval=interval, period=period, progress=False)
                if df is None or df.empty:
                    continue
                # do NOT strip tz; if index naive, we'll assume UTC later
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
        # fallback to naive subtraction
        age_secs = (now_manila.replace(tzinfo=None) - pd.to_datetime(fetched_last_ts_manila).replace(tzinfo=None)).total_seconds()

    # determine staleness: allow tolerance of 1.5 intervals
    interval_seconds = 3600 if interval == "60m" else 86400
    tolerance = interval_seconds * 1.5
    data_is_stale = age_secs > tolerance

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
            # ensure 1D series passed to ta functions
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

    # Live Market View tab
    with tab1:
        st.subheader(f"Live Market View — {ticker}")
        st.write(f"Fetched latest timestamp (converted to Manila): **{fetched_last_ts_manila}**")
        st.write(f"Real-time clock now (Asia/Manila): **{now_manila.strftime('%Y-%m-%d %H:%M:%S %Z')}**")
        st.write(f"Real next-interval target (clock-based): **{real_next_time}**")
        if data_is_stale:
            st.warning("Fetched market data is stale relative to the real-time clock. Predictions will use the latest available data but it may lag real-time.")

        # Candlestick
        df_plot = raw[ticker].copy().dropna()
        close_series = df_plot["Close"].squeeze() if isinstance(df_plot["Close"], pd.DataFrame) else df_plot["Close"]
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot["Open"], high=df_plot["High"], low=df_plot["Low"], close=df_plot["Close"], name=f"{ticker}"))
        if "Close" in df_plot.columns:
            sma20 = close_series.rolling(20).mean()
            fig.add_trace(go.Scatter(x=df_plot.index, y=sma20, mode="lines", name="SMA20", line=dict(width=1)))
        fig.update_layout(height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # Indicators
        st.markdown("**Indicators used:** RSI (14), MACD")
        try:
            rsi = ta.momentum.RSIIndicator(close_series, window=14).rsi()
            macd = ta.trend.MACD(close_series)
            macd_line = macd.macd(); macd_signal = macd.macd_signal()
            fig2 = go.Figure(); fig2.add_trace(go.Scatter(x=df_plot.index, y=rsi, name="RSI(14)")); fig2.update_layout(height=250)
            st.plotly_chart(fig2, use_container_width=True)
            fig3 = go.Figure(); fig3.add_trace(go.Scatter(x=df_plot.index, y=macd_line, name="MACD")); fig3.add_trace(go.Scatter(x=df_plot.index, y=macd_signal, name="MACD sig")); fig3.update_layout(height=250)
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.warning(f"Indicator plotting failed: {e}")

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
            st.warning("Fetched market data is stale relative to the real-time clock. Predictions used latest available data but it may lag.")

        # show predictions table
        if results:
            out = pd.DataFrame(results)[["model", "label", "confidence", "suggestion", "raw"]]

            # display without colored suggestions
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
                # normalize meta label
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
            # determine a human-friendly period word based on interval
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

        # download
        if results:
            st.download_button("Download per-model predictions CSV", data=pd.DataFrame(results).to_csv(index=False).encode("utf-8"), file_name="predictions_next_interval.csv")
