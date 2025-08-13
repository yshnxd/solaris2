# app.py - SOLARIS — Stock Price Movement Predictor (Live multi-period charts)
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
ticker = ticker.strip().upper()

sequence_length = st.sidebar.number_input("Sequence length (LSTM/CNN)", min_value=3, max_value=240, value=24, step=1)

interval = st.sidebar.selectbox("Interval", ["60m", "1d"], index=0)
period_default = "2d" if interval == "60m" else "90d"
period = st.sidebar.text_input("Period (e.g., 2d for intraday, 90d)", period_default)

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

COLOR_BUY_BG = "#d7e9da"
COLOR_SELL_BG = "#f0e5e6"
COLOR_HOLD_BG = "#f3f4f6"
COLOR_TEXT = "#111111"

# ------------------------------
# Helpers (unchanged)
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
    l = (label or "").strip().lower()
    if l == "up":
        return "Buy"
    if l == "down":
        return "Sell"
    return "Hold"

# Prediction history helpers (unchanged)
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
    try:
        df.to_csv(history_file, index=False)
    except Exception as e:
        st.warning(f"Failed saving history to {history_file}: {e}")

def evaluate_history(history_df, aligned_close_df, current_fetched_ts):
    if history_df is None or history_df.empty:
        return history_df
    if 'evaluated' not in history_df.columns:
        history_df['evaluated'] = False
    history_df['target_time'] = pd.to_datetime(history_df['target_time'], errors='coerce')
    now_ts = pd.to_datetime(current_fetched_ts)
    mask = (~history_df['evaluated'].astype(bool)) & (history_df['target_time'].notna()) & (history_df['target_time'] <= now_ts)
    for idx in history_df[mask].index:
        try:
            row = history_df.loc[idx]
            t_target = pd.to_datetime(row['target_time'])
            tk = str(row.get('ticker', '')).upper()
            if tk not in aligned_close_df.columns:
                continue
            series = aligned_close_df[tk].dropna()
            if series.empty:
                continue
            try:
                pred_price = float(row['pred_price']) if pd.notna(row.get('pred_price')) else float(series.asof(pd.to_datetime(row['fetched_last_ts'])))
            except Exception:
                pred_price = None
            try:
                actual_price = float(series.asof(t_target))
            except Exception:
                actual_price = None
            if pred_price is None or actual_price is None or pred_price == 0:
                continue
            pct = (actual_price - pred_price) / pred_price
            pred_label = str(row.get('predicted_label', '')).strip().lower()
            thr = float(neutral_threshold)
            correct = False
            if pred_label == 'up' and pct > 0:
                correct = True
            elif pred_label == 'down' and pct < 0:
                correct = True
            elif pred_label == 'neutral' and abs(pct) <= thr:
                correct = True
            history_df.at[idx, 'evaluated'] = True
            history_df.at[idx, 'pred_price'] = pred_price
            history_df.at[idx, 'actual_price'] = actual_price
            history_df.at[idx, 'pct_change'] = pct
            history_df.at[idx, 'correct'] = bool(correct)
            history_df.at[idx, 'checked_at'] = pd.Timestamp.now()
        except Exception:
            continue
    return history_df

# ------------------------------
# Tabs
# ------------------------------
tab1, tab2, tab3 = st.tabs(["Live Market View", "Predictions", "Detailed Analysis"])

# ------------------------------
# Run pipeline
# ------------------------------
if run_button:
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
        tz_manila = ZoneInfo("Asia/Manila")
    now_manila = datetime.now(tz_manila)

    try:
        age_secs = (now_manila - fetched_last_ts_manila).total_seconds()
    except Exception:
        age_secs = (now_manila.replace(tzinfo=None) - pd.to_datetime(fetched_last_ts_manila).replace(tzinfo=None)).total_seconds()

    interval_seconds = 3600 if interval == "60m" else 86400
    tolerance = interval_seconds * 1.5
    data_is_stale = age_secs > tolerance

    try:
        history = load_history()
        history = evaluate_history(history, aligned_close, fetched_last_ts_manila)
        save_history(history)
    except Exception as e:
        st.warning(f"History evaluation failed: {e}")

    # Build features
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
    # Live Market View (multi-period, stock-like)
    # ------------------------------
    with tab1:
        st.subheader(f"Live Market View — {ticker}")
        st.write(f"Fetched latest timestamp (converted to Manila): **{fetched_last_ts_manila}**")
        st.write(f"Real-time clock now (Asia/Manila): **{now_manila.strftime('%Y-%m-%d %H:%M:%S %Z')}**")
        st.write(f"Real next-interval target (clock-based): **{real_next_time}**")
        if data_is_stale:
            st.warning("Fetched market data is stale relative to the real-time clock. Predictions will use the latest available data but it may lag real-time.")

        # periods we will show at once
        periods_map = {
            "1d": "1m",
            "5d": "15m",
            "1mo": "60m",
            "6mo": "1d",
            "1y": "1d"
        }

        st.markdown("**Charts (all loaded together)**")
        chart_data = {}
        with st.spinner("Fetching chart data for multiple periods..."):
            tk = yf.Ticker(ticker)
            for p, iv in periods_map.items():
                try:
                    # use history; some combos might return empty depending on ticker
                    dfp = tk.history(period=p, interval=iv, auto_adjust=False, prepost=False)
                    if dfp is None or dfp.empty:
                        chart_data[p] = pd.DataFrame()
                    else:
                        dfp = dfp.dropna(how="all")
                        dfp.index = pd.to_datetime(dfp.index)
                        chart_data[p] = dfp
                except Exception:
                    chart_data[p] = pd.DataFrame()

        # choose main_df for the large chart (prefer 1y -> 6mo -> 1mo -> 5d -> 1d)
        main_df = None
        for prefer in ["1y", "6mo", "1mo", "5d", "1d"]:
            if prefer in chart_data and not chart_data[prefer].empty:
                main_df = chart_data[prefer]
                break
        if main_df is None:
            st.info("No chart data available for any of the periods.")
            chart_df = pd.DataFrame()  # fallback
        else:
            # build stock-like chart with range selector
            fig = go.Figure()

            fig.add_trace(go.Candlestick(
                x=main_df.index,
                open=main_df['Open'],
                high=main_df['High'],
                low=main_df['Low'],
                close=main_df['Close'],
                name=ticker
            ))

            # SMA (visual)
            try:
                if len(main_df['Close']) >= 20:
                    sma20 = main_df['Close'].rolling(20).mean()
                    fig.add_trace(go.Scatter(x=main_df.index, y=sma20, mode='lines', name='SMA20', line=dict(width=1)))
            except Exception:
                pass

            # volume bars on second axis
            try:
                fig.add_trace(go.Bar(x=main_df.index, y=main_df['Volume'], name='Volume', marker=dict(opacity=0.4), yaxis='y2'))
            except Exception:
                pass

            # current price annotation (last close)
            try:
                last_price = float(main_df['Close'].iloc[-1])
                last_time = main_df.index[-1]
                fig.add_hline(y=last_price, line_dash="dash", line_color="black", annotation_text=f"Last: {last_price:.2f}", annotation_position="top left")
            except Exception:
                pass

            # range selector buttons
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1d", step="day", stepmode="backward"),
                            dict(count=5, label="5d", step="day", stepmode="backward"),
                            dict(count=1, label="1mo", step="month", stepmode="backward"),
                            dict(count=6, label="6mo", step="month", stepmode="backward"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type="date"
                ),
                yaxis=dict(domain=[0, 0.75]),
                yaxis2=dict(domain=[0.0, 0.2], anchor="x", overlaying="y", side="right", showgrid=False),
                height=650,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig, use_container_width=True)

            # show small snapshots for every period in a row (candles mini-charts)
            st.markdown("**Snapshots:**")
            cols = st.columns(len(periods_map))
            for i, (p, iv) in enumerate(periods_map.items()):
                with cols[i]:
                    st.write(p)
                    dfp = chart_data.get(p, pd.DataFrame())
                    if dfp is None or dfp.empty:
                        st.write("No data")
                    else:
                        try:
                            fig_small = go.Figure()
                            fig_small.add_trace(go.Candlestick(
                                x=dfp.index, open=dfp['Open'], high=dfp['High'], low=dfp['Low'], close=dfp['Close'], showlegend=False
                            ))
                            # remove rangeslider for small
                            fig_small.update_layout(height=220, margin=dict(t=10,b=10,l=10,r=10), xaxis_rangeslider_visible=False)
                            st.plotly_chart(fig_small, use_container_width=True)
                        except Exception:
                            st.write("Chart failed")

            # set chart_df used later in analysis: prefer 1d intraday if present
            chart_df = chart_data.get("1d") if ("1d" in chart_data and not chart_data["1d"].empty) else main_df

    # ------------------------------
    # Load models and rest of pipeline (unchanged)
    # ------------------------------
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
                    raw_pred = mod.predict(X_in.reshape((1,-1)))
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

    # ------------------------------
    # Predictions tab and remainder (unchanged; uses chart_df set above)
    # ------------------------------
    with tab2:
        st.subheader("Predictions — next interval (real-time)")
        st.write(f"Ticker: **{ticker}**   •   Fetched latest (Manila): **{fetched_last_ts_manila}**")
        st.write(f"Real-time clock now (Asia/Manila): **{now_manila.strftime('%Y-%m-%d %H:%M:%S %Z')}**")
        st.write(f"Real next-interval target (clock-based): **{real_next_time}**")
        if data_is_stale:
            st.warning("Fetched market data is stale relative to the real-time clock. Predictions used latest available data but it may lag.")

        if results:
            out = pd.DataFrame(results)[["model", "label", "confidence", "suggestion", "raw"]]
            try:
                st.dataframe(out)
            except Exception:
                st.dataframe(out)
        else:
            st.warning("No predictions produced.")

        # conclusive/meta logic unchanged...
        # (snipped here because unchanged for brevity in the explanation — keep the same code you already have)
        # ... but ensure you keep the conclusive block and history-append logic from your existing file,
        # and note that `chart_df` variable is available (1d or main_df) for Detailed Analysis.

    # ------------------------------
    # Detailed Analysis tab (unchanged)
    # ------------------------------
    # Detailed Analysis tab
    # ------------------------------
    with tab3:
        st.subheader("Detailed Analysis)

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
