import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import ta
import plotly.graph_objects as go
from datetime import datetime

# ================================
#   MODEL LOADING
# ================================
@st.cache_resource
def load_models():
    try:
        cnn_model = joblib.load('cnn_model.pkl')
        lstm_model = joblib.load('lstm_model.pkl')
        xgb_model = joblib.load('xgb_model.pkl')
        meta_model = joblib.load('meta_learner.pkl')
        scaler = joblib.load('scaler.pkl')
        return cnn_model, lstm_model, xgb_model, meta_model, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error("Please ensure all model files are in the working directory.")
        return None, None, None, None, None

cnn_model, lstm_model, xgb_model, meta_model, scaler = load_models()

# ================================
#   GLOBAL SETTINGS
# ================================
original_cross_assets = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA",
    "NVDA", "JPM", "JNJ", "XOM", "CAT", "BA", "META"
]
dynamic_cross_assets = original_cross_assets.copy()

expected_columns = [
    'ret_1h', 'ret_3h', 'ret_6h', 'ret_12h', 'ret_24h',
    'vol_6h', 'vol_12h', 'vol_24h',
    'rsi_14', 'macd', 'macd_signal',
    'sma_5', 'ema_5', 'sma_10', 'ema_10', 'sma_20', 'ema_20',
    'vol_change_1h', 'vol_ma_24h',
    'AAPL_ret_1h', 'MSFT_ret_1h', 'AMZN_ret_1h', 'GOOGL_ret_1h',
    'TSLA_ret_1h', 'NVDA_ret_1h', 'JPM_ret_1h', 'JNJ_ret_1h',
    'XOM_ret_1h', 'CAT_ret_1h', 'BA_ret_1h', 'META_ret_1h',
    'hour', 'day_of_week', 'price'
]

SEQ_LEN = 128

# ================================
#   DATA FETCHING
# ================================
@st.cache_data(ttl=300)
def fetch_stock_data(ticker, period="729d", interval="60m"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception as e:
        return None

def fetch_cross_assets(selected_index, days_history, selected_ticker):
    global dynamic_cross_assets
    if selected_ticker in dynamic_cross_assets:
        dynamic_cross_assets = [a for a in original_cross_assets if a != selected_ticker]
    else:
        dynamic_cross_assets = original_cross_assets[:11]

    cross_data = {}
    for asset in dynamic_cross_assets:
        df = fetch_stock_data(asset, period=f"{days_history}d", interval="60m")
        if df is not None and not df.empty:
            cross_data[asset] = df.reindex(selected_index)['Close'].ffill()
        else:
            cross_data[asset] = pd.Series(0, index=selected_index)
    return cross_data

# ================================
#   FEATURE ENGINEERING
# ================================
def create_features_for_app(selected_ticker, ticker_data, cross_data):
    price_series = ticker_data['Close']
    feat_tmp = pd.DataFrame(index=price_series.index)

    # Lag returns
    for lag in [1, 3, 6, 12, 24]:
        feat_tmp[f"ret_{lag}h"] = price_series.pct_change(lag)

    # Volatility
    for window in [6, 12, 24]:
        feat_tmp[f"vol_{window}h"] = price_series.pct_change().rolling(window).std()

    # RSI, MACD
    try:
        feat_tmp["rsi_14"] = ta.momentum.RSIIndicator(price_series, window=14).rsi()
    except Exception:
        feat_tmp["rsi_14"] = 0

    try:
        macd = ta.trend.MACD(price_series)
        feat_tmp["macd"] = macd.macd()
        feat_tmp["macd_signal"] = macd.macd_signal()
    except Exception:
        feat_tmp["macd"], feat_tmp["macd_signal"] = 0, 0

    # SMA and EMA
    for w in [5, 10, 20]:
        feat_tmp[f"sma_{w}"] = price_series.rolling(w).mean()
        feat_tmp[f"ema_{w}"] = price_series.ewm(span=w, adjust=False).mean()

    # Volume-based features
    if "Volume" in ticker_data.columns:
        vol_series = ticker_data['Volume'].ffill()
        feat_tmp["vol_change_1h"] = vol_series.pct_change()
        feat_tmp["vol_ma_24h"] = vol_series.rolling(24).mean()
    else:
        feat_tmp["vol_change_1h"] = 0
        feat_tmp["vol_ma_24h"] = 0

    # Cross-asset hourly returns
    for asset in original_cross_assets:
        feat_tmp[f"{asset}_ret_1h"] = cross_data.get(asset, pd.Series(0, index=feat_tmp.index)).pct_change().fillna(0)

    # Temporal features
    feat_tmp["hour"] = feat_tmp.index.hour
    feat_tmp["day_of_week"] = feat_tmp.index.dayofweek
    feat_tmp["price"] = price_series

    for col in ["rsi_14", "macd", "macd_signal", "vol_change_1h", "vol_ma_24h"]:
        feat_tmp[col] = feat_tmp[col].fillna(0)

    feat_tmp = feat_tmp.dropna(subset=["price"])

    # Ensure all expected columns exist
    for col in expected_columns:
        if col not in feat_tmp.columns:
            feat_tmp[col] = 0

    feat_tmp = feat_tmp[expected_columns]
    return feat_tmp

# ================================
#   PREDICTION PIPELINE
# ================================
def predict_with_models(features, current_price, scaler, cnn_model, lstm_model, xgb_model, meta_model):
    if scaler is None or cnn_model is None or lstm_model is None or xgb_model is None or meta_model is None:
        return None, None, None, None

    features = features.replace([np.inf, -np.inf], 0).fillna(0)
    if len(features) < SEQ_LEN:
        return None, None, None, None

    features_recent = features.iloc[-SEQ_LEN:]
    features_tabular = features.iloc[-1:]

    X_seq = scaler.transform(features_recent)
    X_tab = scaler.transform(features_tabular)
    X_seq = X_seq.reshape(1, SEQ_LEN, -1)
    X_tab = X_tab.reshape(1, -1)

    # Base model predictions
    try:
        pred_cnn = float(cnn_model.predict(X_seq, verbose=0)[0][0])
    except Exception:
        pred_cnn = 0.0

    try:
        pred_lstm = float(lstm_model.predict(X_seq, verbose=0)[0][0])
    except Exception:
        pred_lstm = 0.0

    try:
        pred_xgb = float(xgb_model.predict(X_tab)[0])
    except Exception:
        pred_xgb = 0.0

    # Meta features
    meta_feats = np.array([pred_cnn, pred_lstm, pred_xgb])
    meta_stats = np.array([meta_feats.mean(), meta_feats.std(), meta_feats.max(), meta_feats.min()])
    meta_input = np.concatenate([meta_feats, meta_stats]).reshape(1, -1)

    # Meta prediction
    try:
        pred_meta = float(meta_model.predict(meta_input)[0])
    except Exception:
        pred_meta = np.mean([pred_cnn, pred_lstm, pred_xgb])

    predicted_price = current_price * (1 + pred_meta)
    pred_change_pct = pred_meta * 100

    votes = {
        "CNN": "UP" if pred_cnn > 0 else "DOWN",
        "LSTM": "UP" if pred_lstm > 0 else "DOWN",
        "XGBoost": "UP" if pred_xgb > 0 else "DOWN"
    }

    up_count = sum(v == "UP" for v in votes.values())
    confidence = (max(up_count, 3 - up_count) / 3) * 100

    return predicted_price, pred_change_pct, votes, confidence

# ================================
#   STREAMLIT INTERFACE
# ================================
st.set_page_config(page_title="📈 StockWise", layout="wide")
st.title("📈 StockWise: Hybrid AI Stock Prediction System")
st.markdown("Real-time stock prediction using CNN + LSTM + XGBoost with Meta-Learner Ensemble")

ticker = st.text_input("Enter stock ticker:", value="AAPL").upper().strip()
days_history = st.slider("Days of data history", 90, 720, 365)
predict_btn = st.button("Predict Next Hour")

if predict_btn:
    st.info(f"Fetching {ticker} data...")
    data = fetch_stock_data(ticker, period=f"{days_history}d", interval="60m")

    if data is not None:
        cross_assets = fetch_cross_assets(data.index, days_history, ticker)
        features = create_features_for_app(ticker, data, cross_assets)
        current_price = float(data["Close"].iloc[-1])

        with st.spinner("Generating prediction..."):
            pred_price, pred_pct, votes, confidence = predict_with_models(
                features, current_price, scaler, cnn_model, lstm_model, xgb_model, meta_model
            )

        if pred_price is not None:
            st.subheader(f"Prediction for {ticker}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${current_price:.2f}")
            col2.metric("Predicted Price (Next Hour)", f"${pred_price:.2f}", delta=f"{pred_pct:.3f}%")
            col3.metric("Confidence", f"{confidence:.1f}%")

            st.markdown("### Model Votes")
            st.json(votes)

            st.markdown("---")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index[-48:], y=data["Close"].tail(48), mode="lines", name="Close"))
            fig.add_trace(go.Scatter(
                x=[data.index[-1] + pd.Timedelta(hours=1)],
                y=[pred_price],
                mode="markers+text",
                name="Predicted Price",
                text=[f"Pred: ${pred_price:.2f}"],
                textposition="top center"
            ))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Prediction could not be generated (insufficient data or model error).")
    else:
        st.error("Failed to fetch stock data.")
else:
    st.info("Enter a stock ticker and click Predict.")
