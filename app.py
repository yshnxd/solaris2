# app.py - StockWise Streamlit deployment (uses .pkl models for CNN, LSTM, XGB, Meta)
# Save this file next to: cnn_model.pkl, lstm_model.pkl, xgb_model.pkl, meta_learner.pkl, scaler.pkl

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import joblib
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings("ignore")

# ------- Configuration -------
SEQ_LEN = 128
HOURS_TO_FETCH = 200
CROSS_ASSET_TICKERS = ["AAPL","MSFT","AMZN","GOOGL","TSLA","NVDA","JPM","JNJ","XOM","CAT","BA","META"]
FEATURE_COLUMNS = [
    "ret_1h","ret_3h","ret_6h","ret_12h","ret_24h",
    "vol_6h","vol_12h","vol_24h",
    "rsi_14","macd","macd_signal",
    "sma_5","sma_10","sma_20","ema_5","ema_10","ema_20",
    "vol_change_1h","vol_ma_24h",
    "hour","day_of_week"
]
MODEL_FILES = {
    "cnn": "cnn_model.pkl",
    "lstm": "lstm_model.pkl",
    "xgb": "xgb_model.pkl",
    "meta": "meta_learner.pkl",
    "scaler": "scaler.pkl"
}

# ------- Feature engineering functions -------
def fetch_hourly_data(ticker, period_hours=HOURS_TO_FETCH):
    days = max(3, int(np.ceil(period_hours / 24.0) + 1))
    df = yf.download(tickers=ticker, period=f"{days}d", interval="60m", progress=False, threads=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}.")
    df = df.dropna(how="all").tail(period_hours)
    df.index = pd.to_datetime(df.index).tz_convert(None)
    return df

def add_technical_features(df):
    df = df.copy()
    close = df['Close']
    df['ret_1h'] = close.pct_change(1)
    df['ret_3h'] = close.pct_change(3)
    df['ret_6h'] = close.pct_change(6)
    df['ret_12h'] = close.pct_change(12)
    df['ret_24h'] = close.pct_change(24)
    for w, name in [(6,'vol_6h'),(12,'vol_12h'),(24,'vol_24h')]:
        df[name] = df['ret_1h'].rolling(window=w, min_periods=1).std()
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean().replace(0,1e-8)
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['sma_5'] = close.rolling(5, min_periods=1).mean()
    df['sma_10'] = close.rolling(10, min_periods=1).mean()
    df['sma_20'] = close.rolling(20, min_periods=1).mean()
    df['ema_5'] = close.ewm(span=5, adjust=False).mean()
    df['ema_10'] = close.ewm(span=10, adjust=False).mean()
    df['ema_20'] = close.ewm(span=20, adjust=False).mean()
    df['vol_change_1h'] = df['Volume'].pct_change(1)
    df['vol_ma_24h'] = df['Volume'].rolling(window=24, min_periods=1).mean()
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

def compute_cross_asset_returns(main_index, symbols):
    out = pd.DataFrame(index=main_index)
    for s in symbols:
        try:
            df = fetch_hourly_data(s, period_hours=HOURS_TO_FETCH)
            df = df[['Close']].rename(columns={'Close': f'{s}_Close'})
            df = df.reindex(main_index).ffill().bfill()
            out[f"{s}_ret_1h"] = df[f'{s}_Close'].pct_change().fillna(0)
        except Exception:
            out[f"{s}_ret_1h"] = 0.0
    return out

def build_feature_matrix(ticker_df, cross_df):
    df = pd.concat([ticker_df, cross_df], axis=1)
    cross_cols = list(cross_df.columns)
    cols = [c for c in FEATURE_COLUMNS if c in df.columns] + cross_cols
    X = df[cols].replace([np.inf,-np.inf],np.nan).fillna(method='ffill').fillna(0.0)
    return X, cols

def create_seq_inputs(X_np, seq_len=SEQ_LEN):
    if X_np.shape[0] < seq_len:
        raise RuntimeError(f"Not enough data for sequence length {seq_len}")
    seq = X_np[-seq_len:]
    return seq.reshape(1, seq_len, X_np.shape[1])

# ------- Model loading -------
@st.cache_resource
def load_models(base_dir="."):
    models = {}
    for k,v in MODEL_FILES.items():
        path = os.path.join(base_dir, v)
        models[k] = joblib.load(path) if os.path.exists(path) else None
    return models

# ------- Prediction logic -------
def predict_next_hour(ticker, models):
    df = fetch_hourly_data(ticker, period_hours=HOURS_TO_FETCH)
    df_feat = add_technical_features(df)
    cross_df = compute_cross_asset_returns(df_feat.index, CROSS_ASSET_TICKERS)
    X_df, used_cols = build_feature_matrix(df_feat, cross_df)

    scaler = models['scaler']
    if scaler is None:
        raise RuntimeError("Missing scaler.pkl")
    X_scaled = scaler.transform(X_df)
    seq_input = create_seq_inputs(X_scaled)
    last_tab = X_scaled[-1].reshape(1, -1)

    preds = {}
    for m in ['cnn','lstm','xgb']:
        if models[m] is not None:
            preds[m] = float(models[m].predict(seq_input if m in ['cnn','lstm'] else last_tab)[0])
        else:
            preds[m] = np.nan

    meta_row = np.array([[preds['cnn'], preds['lstm'], preds['xgb']]])
    meta_model = models['meta']
    if meta_model is None:
        raise RuntimeError("Missing meta_learner.pkl")
    meta_pred = float(meta_model.predict(meta_row)[0])

    last_price = float(df['Close'].iloc[-1])
    pred_price = last_price * (1.0 + meta_pred)

    return {
        "timestamp": df.index[-1],
        "last_price": last_price,
        "pred_return": meta_pred,
        "pred_price": pred_price,
        "base_preds": preds
    }, df

# ------- Streamlit Interface -------
st.set_page_config(page_title="📈 StockWise", layout="wide")
st.title("📈 StockWise: Hybrid AI Stock Prediction System")

with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    run_button = st.button("Predict")
    st.markdown("---")
    st.caption("Ensure all .pkl model files are in the same folder.")

models = load_models(".")

if run_button:
    st.info(f"Running StockWise for {ticker} ...")
    try:
        with st.spinner("Generating prediction..."):
            result, df = predict_next_hour(ticker, models)

        st.subheader(f"Prediction for {ticker} — {result['timestamp']}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Last Price", f"${result['last_price']:.2f}")
        col2.metric("Next-Hour Forecast", f"${result['pred_price']:.2f}", delta=f"{result['pred_return']*100:.3f}%")
        col3.metric("Consensus", f"{np.std(list(result['base_preds'].values())):.3f} σ")

        st.markdown("---")
        st.subheader("Recent Prices & Predicted Next-Hour Point")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
        next_time = df.index[-1] + pd.Timedelta(hours=1)
        fig.add_trace(go.Scatter(x=[next_time], y=[result['pred_price']], mode='markers+text',
                                 text=[f"Pred: ${result['pred_price']:.2f}"], textposition="top center"))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Model Outputs"):
            st.json(result['base_preds'])

        st.success("Prediction completed successfully.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
else:
    st.info("Enter a ticker symbol and click Predict.")
