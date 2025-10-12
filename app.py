# app.py - StockWise Streamlit deployment (implements methodology from uploaded paper)
# Save this file next to your model artifacts: cnn_model.h5, lstm_model.h5, xgb_model.pkl, meta_learner.pkl, scaler.pkl

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import joblib
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import xgboost as xgb
import os
import warnings
warnings.filterwarnings("ignore")

# ------- Configuration (tweakable) -------
SEQ_LEN = 128                     # sequence length used in training (per paper)
HOURS_TO_FETCH = 200              # fetch this many hours to ensure sequence windows available
CROSS_ASSET_TICKERS = ["AAPL","MSFT","AMZN","GOOGL","TSLA","NVDA","JPM","JNJ","XOM","CAT","BA","META"]
FEATURE_COLUMNS = [
    # lag returns
    "ret_1h","ret_3h","ret_6h","ret_12h","ret_24h",
    # volatility
    "vol_6h","vol_12h","vol_24h",
    # momentum / technicals
    "rsi_14","macd","macd_signal",
    # moving averages
    "sma_5","sma_10","sma_20","ema_5","ema_10","ema_20",
    # volume
    "vol_change_1h","vol_ma_24h",
    # cross-asset one-hour returns will be appended dynamically
    "hour","day_of_week"
]
MODEL_FILES = {
    "cnn": "cnn_model.h5",
    "lstm": "lstm_model.h5",
    "xgb": "xgb_model.pkl",
    "meta": "meta_learner.pkl",
    "scaler": "scaler.pkl"
}

# ------- Utility functions for feature engineering (mirrors paper) -------
def fetch_hourly_data(ticker, period_hours=HOURS_TO_FETCH):
    """
    Use yfinance to fetch hourly OHLCV. We request a period sufficient to cover required sequences.
    Return DataFrame with DatetimeIndex (UTC).
    """
    # yfinance supports period strings; approximate hours -> days
    days = max(3, int(np.ceil(period_hours / 24.0) + 1))
    try:
        df = yf.download(tickers=ticker, period=f"{days}d", interval="60m", progress=False, threads=False)
    except Exception as e:
        raise RuntimeError(f"yfinance download error for {ticker}: {e}")
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}.")
    # Keep only last `period_hours` rows (in case API returns finer)
    df = df.dropna(how="all")
    df = df.tail(period_hours)
    df.index = pd.to_datetime(df.index).tz_convert(None)
    return df

def add_technical_features(df):
    """
    Given OHLCV datframe, compute features according to methodology:
    - lag returns (1,3,6,12,24)
    - rolling volatility (6,12,24)
    - RSI(14), MACD (12,26) and MACD signal (9)
    - SMA/EMA 5/10/20
    - volume change 1h, volume MA 24h
    - hour, day_of_week
    """
    df = df.copy()
    # close series
    close = df['Close']
    # lag returns
    df['ret_1h'] = close.pct_change(1)
    df['ret_3h'] = close.pct_change(3)
    df['ret_6h'] = close.pct_change(6)
    df['ret_12h'] = close.pct_change(12)
    df['ret_24h'] = close.pct_change(24)
    # rolling vol (std of returns)
    for w, name in [(6,'vol_6h'),(12,'vol_12h'),(24,'vol_24h')]:
        df[name] = df['ret_1h'].rolling(window=w, min_periods=1).std()
    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean().replace(0,1e-8)
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    # MACD (12,26) and signal (9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    # SMA & EMA
    df['sma_5'] = close.rolling(5, min_periods=1).mean()
    df['sma_10'] = close.rolling(10, min_periods=1).mean()
    df['sma_20'] = close.rolling(20, min_periods=1).mean()
    df['ema_5'] = close.ewm(span=5, adjust=False).mean()
    df['ema_10'] = close.ewm(span=10, adjust=False).mean()
    df['ema_20'] = close.ewm(span=20, adjust=False).mean()
    # volume features
    df['vol_change_1h'] = df['Volume'].pct_change(1)
    df['vol_ma_24h'] = df['Volume'].rolling(window=24, min_periods=1).mean()
    # calendar
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    # forward-fill / fillna (safe, but avoid leaking future info by using only past for sequences)
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

def compute_cross_asset_returns(main_index_ts, symbols):
    """
    Fetch the one-hour returns for cross-asset tickers and align to index of main ticker.
    Returns a DataFrame with columns like 'AAPL_ret_1h' etc.
    """
    out = pd.DataFrame(index=main_index_ts)
    for s in symbols:
        try:
            df = fetch_hourly_data(s, period_hours=HOURS_TO_FETCH)
            df = df[['Close']].rename(columns={'Close': f'{s}_Close'})
            df.index = pd.to_datetime(df.index).tz_convert(None)
            # align to main index by reindexing (forward-fill)
            df = df.reindex(main_index_ts).ffill().bfill()
            out[f"{s}_ret_1h"] = df[f'{s}_Close'].pct_change().fillna(0)
        except Exception:
            out[f"{s}_ret_1h"] = 0.0
    return out

def build_feature_matrix(ticker_df, cross_returns_df):
    """
    Given feature-engineered df and cross returns, compose final feature matrix row-aligned.
    """
    df = ticker_df.copy()
    # append cross-asset returns
    df = pd.concat([df, cross_returns_df], axis=1)
    # ensure columns consistent with training (drop extras)
    # Compose final ordered column list: base FEATURE_COLUMNS + cross asset names
    cross_cols = list(cross_returns_df.columns)
    cols = [c for c in FEATURE_COLUMNS if c in df.columns] + cross_cols
    X = df[cols].copy()
    # replace inf/nan
    X = X.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0.0)
    return X, cols

def create_seq_inputs(X_np, seq_len=SEQ_LEN):
    """
    Prepare last sequence windows for CNN/LSTM (shape: (1, seq_len, n_features))
    This returns the latest available full sequence.
    """
    if X_np.shape[0] < seq_len:
        raise RuntimeError(f"Not enough historical rows ({X_np.shape[0]}) for sequence length {seq_len}")
    seq = X_np[-seq_len:]
    return seq.reshape(1, seq_len, X_np.shape[1])

# ------- Model loading -------
@st.cache_resource
def load_models(base_dir="."):
    models = {}
    # CNN & LSTM
    if os.path.exists(os.path.join(base_dir, MODEL_FILES['cnn'])):
        models['cnn'] = tf.keras.models.load_model(os.path.join(base_dir, MODEL_FILES['cnn']))
    else:
        models['cnn'] = None
    if os.path.exists(os.path.join(base_dir, MODEL_FILES['lstm'])):
        models['lstm'] = tf.keras.models.load_model(os.path.join(base_dir, MODEL_FILES['lstm']))
    else:
        models['lstm'] = None
    # XGB and meta
    models['xgb'] = joblib.load(os.path.join(base_dir, MODEL_FILES['xgb'])) if os.path.exists(os.path.join(base_dir, MODEL_FILES['xgb'])) else None
    models['meta'] = joblib.load(os.path.join(base_dir, MODEL_FILES['meta'])) if os.path.exists(os.path.join(base_dir, MODEL_FILES['meta'])) else None
    models['scaler'] = joblib.load(os.path.join(base_dir, MODEL_FILES['scaler'])) if os.path.exists(os.path.join(base_dir, MODEL_FILES['scaler'])) else None
    return models

# ------- Prediction pipeline (follows paper methodology) -------
def predict_next_hour(ticker, models):
    """
    Full pipeline:
      - fetch hourly data for ticker
      - compute features & cross-asset returns
      - scale tabular features with scaler
      - create sequence inputs for CNN/LSTM
      - get base model predictions
      - stack and feed meta-learner
      - return predicted next-hour return and predicted price
    """
    # 1) get hourly data for main ticker
    df = fetch_hourly_data(ticker, period_hours=HOURS_TO_FETCH)
    df_feat = add_technical_features(df)
    # 2) cross asset returns aligned to main df index
    cross_df = compute_cross_asset_returns(df_feat.index, CROSS_ASSET_TICKERS)
    # 3) build feature matrix
    X_df, used_cols = build_feature_matrix(df_feat, cross_df)
    # 4) prepare inputs for tabular model (XGBoost) - use last row as features for current time
    X_tab = X_df.values
    last_tab = X_tab[-1].reshape(1, -1)
    # 5) scale (scaler fitted on training data only per methodology)
    if models['scaler'] is None:
        raise RuntimeError("scaler.pkl not found. Save and place scaler.pkl in app directory.")
    scaler = models['scaler']
    last_tab_scaled = scaler.transform(last_tab)
    X_tab_col_names = used_cols
    # 6) prepare sequences for CNN/LSTM (the same feature order and scaler was used during training for sequences)
    # For sequences we scale entire matrix before sequencing, preserving temporal scaling applied in training pipeline
    X_all_scaled = scaler.transform(X_tab)  # shape (N, n_features)
    seq_input = create_seq_inputs(X_all_scaled, seq_len=SEQ_LEN)  # shape (1, SEQ_LEN, n_features)
    # 7) Base model predictions
    preds = {}
    # CNN - expects shape (batch, seq_len, n_features)
    if models['cnn'] is not None:
        try:
            preds['cnn'] = float(models['cnn'].predict(seq_input, verbose=0).ravel()[0])
        except Exception as e:
            preds['cnn'] = np.nan
    else:
        preds['cnn'] = np.nan
    # LSTM
    if models['lstm'] is not None:
        try:
            preds['lstm'] = float(models['lstm'].predict(seq_input, verbose=0).ravel()[0])
        except Exception as e:
            preds['lstm'] = np.nan
    else:
        preds['lstm'] = np.nan
    # XGBoost - uses tabular last row (scaled)
    if models['xgb'] is not None:
        try:
            preds['xgb'] = float(models['xgb'].predict(last_tab_scaled)[0])
        except Exception as e:
            preds['xgb'] = np.nan
    else:
        preds['xgb'] = np.nan
    # 8) meta-learner: form meta-row [cnn_pred, lstm_pred, xgb_pred] (+ optionally extra features; per paper we include some extras)
    meta_row = np.array([[preds['cnn'] if not np.isnan(preds['cnn']) else 0.0,
                          preds['lstm'] if not np.isnan(preds['lstm']) else 0.0,
                          preds['xgb'] if not np.isnan(preds['xgb']) else 0.0]])
    # If meta model expects more features (paper sometimes augments with original extras),
    # we attempt to append the same last_tab features if meta model input dimension >3.
    meta_model = models['meta']
    if meta_model is None:
        raise RuntimeError("meta_learner.pkl not found.")
    # If meta expects additional features, concatenate last_tab_scaled
    try:
        expected_dim = getattr(meta_model, "n_features_in_", None)
        if expected_dim is None:
            # scikit-learn HGBR has n_features_in_
            expected_dim = meta_model.coef_.shape[0] if hasattr(meta_model, "coef_") else 3
    except Exception:
        expected_dim = 3
    if expected_dim > meta_row.shape[1]:
        # append subset of scaled tab features to match size (paper augmented meta features)
        to_append = last_tab_scaled[0, :expected_dim - meta_row.shape[1]]
        meta_row = np.hstack([meta_row, to_append.reshape(1, -1)])
    # final meta predict (next-hour return)
    meta_pred = float(meta_model.predict(meta_row)[0])
    # convert to predicted price
    last_price = float(df['Close'].iloc[-1])
    pred_price = last_price * (1.0 + meta_pred)
    # build detailed response
    result = {
        "timestamp": df.index[-1],
        "last_price": last_price,
        "pred_return": meta_pred,
        "pred_price": pred_price,
        "base_preds": preds,
        "meta_row": meta_row,
        "features_used": X_tab_col_names
    }
    return result, df, X_df

# ------- Streamlit UI -------
st.set_page_config(page_title="📈 StockWise: Hybrid AI Stock Prediction System", layout="wide")
st.title("📈 StockWise: Hybrid AI Stock Prediction System")
st.markdown("Deployed hybrid model (CNN + LSTM + XGBoost) with stacking meta-learner — follows methodology described in the research paper.")

with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker symbol", value="AAPL").upper().strip()
    run_button = st.button("Predict")
    st.markdown("---")
    st.mark.caption("Models & scaler must be present in app directory:")
    for k,v in MODEL_FILES.items():
        st.caption(f"{k}: {v}")

# Load models once
models = load_models(".")
if st.button("Check model status"):
    st.write({k: ("OK" if models[k] is not None else "MISSING") for k in models})

if run_button:
    st.info(f"Fetching data and running inference for {ticker} ...")
    try:
        with st.spinner("Fetching data and computing features..."):
            result, full_df, features_df = predict_next_hour(ticker, models)
        # Show top-line results
        st.subheader(f"Prediction for {ticker} at {result['timestamp']}")
        col1, col2, col3 = st.columns([1.2,1,1])
        col1.metric("Last Price (USD)", f"${result['last_price']:.2f}")
        col2.metric("Predicted Next-Hour Price", f"${result['pred_price']:.2f}", delta=f"{result['pred_return']*100:.3f}%")
        # simple confidence heuristic: spread between base preds
        base_vals = np.array([v for v in result['base_preds'].values() if not np.isnan(v)])
        if base_vals.size>0:
            conf = 1.0 / (1.0 + np.std(base_vals))  # heuristic: lower spread => higher confidence
            col3.metric("Model Consensus (heuristic)", f"{conf:.2f}")
        else:
            col3.metric("Model Consensus (heuristic)", "N/A")
        st.markdown("---")
        # Plot recent prices and predicted point
        st.subheader("Price chart (recent) with predicted next-hour price")
        hist = full_df[['Close']].copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close'))
        # predicted point at next hour index
        next_time = hist.index[-1] + pd.Timedelta(hours=1)
        fig.add_trace(go.Scatter(x=[next_time], y=[result['pred_price']], mode='markers+text', name='Predicted Price',
                                 marker=dict(size=10), text=[f"Pred: ${result['pred_price']:.2f}"], textposition="top center"))
        fig.update_layout(height=420, margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(fig, use_container_width=True)
        # RSI and MACD
        st.subheader("Technical indicators (last 48 hours)")
        ind_df = features_df.tail(48)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=ind_df.index, y=ind_df['rsi_14'], name='RSI(14)'))
        fig2.update_layout(height=250, margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig2, use_container_width=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=ind_df.index, y=ind_df['macd'], name='MACD'))
        fig3.add_trace(go.Scatter(x=ind_df.index, y=ind_df['macd_signal'], name='MACD Signal'))
        fig3.update_layout(height=250, margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig3, use_container_width=True)
        # expandable: show base model preds and raw meta row
        with st.expander("Intermediate model outputs and meta features"):
            st.write("Base model predictions (OOF-derived stacking inputs):")
            st.json(result['base_preds'])
            st.write("Meta input row (what was fed to meta-learner):")
            st.write(result['meta_row'])
            st.write("Feature columns used for tabular XGB input:")
            st.write(result['features_used'])
        # logging: append prediction to CSV log
        log_row = {
            "timestamp": result['timestamp'],
            "ticker": ticker,
            "last_price": result['last_price'],
            "pred_price": result['pred_price'],
            "pred_return": result['pred_return'],
            "cnn_pred": result['base_preds'].get('cnn'),
            "lstm_pred": result['base_preds'].get('lstm'),
            "xgb_pred": result['base_preds'].get('xgb')
        }
        log_df = pd.DataFrame([log_row])
        logfile = "predictions_log.csv"
        if os.path.exists(logfile):
            log_df.to_csv(logfile, mode='a', header=False, index=False)
        else:
            log_df.to_csv(logfile, index=False)
        st.success("Prediction completed and logged.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)
else:
    st.info("Enter a ticker and press Predict to run the StockWise pipeline.")

# Footer: short explanation of stack logic
st.markdown("---")
st.mark.caption("Notes: This app follows the StockWise methodology: CNN & LSTM use sequence windows (128h) derived from the same scaled features; XGBoost uses tabular engineered indicators. Out-of-fold stacking / meta-learner logic is used offline to train the meta model which is loaded here for inference. See paper for full training & OOF details. :contentReference[oaicite:1]{index=1}")
