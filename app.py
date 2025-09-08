import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import ta
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
import json

# Page config
st.set_page_config(
    page_title="SOLARIS - ML Stock Prediction",
    page_icon="📈",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    cnn_model = tf.keras.models.load_model('models/cnn_model.h5')
    lstm_model = tf.keras.models.load_model('models/lstm_model.h5')
    xgb_model = joblib.load('models/xgb_model.pkl')
    meta_model = joblib.load('models/meta_learner.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return cnn_model, lstm_model, xgb_model, meta_model, scaler

# Feature Engineering function
def create_features(df):
    df_feat = pd.DataFrame(index=df.index)
    
    # Price/Return features
    for lag in [1, 3, 6, 12, 24]:
        df_feat[f'ret_{lag}h'] = df['Close'].pct_change(lag)
    
    # Volatility
    for window in [6, 12, 24]:
        df_feat[f'vol_{window}h'] = df['Close'].pct_change().rolling(window).std()
    
    # Technical Indicators
    df_feat['rsi_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    macd = ta.trend.MACD(df['Close'])
    df_feat['macd'] = macd.macd()
    df_feat['macd_signal'] = macd.macd_signal()
    
    # Moving averages
    for w in [5, 10, 20]:
        df_feat[f'sma_{w}'] = df['Close'].rolling(w).mean()
        df_feat[f'ema_{w}'] = df['Close'].ewm(span=w, adjust=False).mean()
    
    # Volume features
    if 'Volume' in df.columns:
        df_feat['vol_change_1h'] = df['Volume'].pct_change()
        df_feat['vol_ma_24h'] = df['Volume'].rolling(24).mean()
    
    # Calendar
    df_feat['hour'] = df.index.hour
    df_feat['day_of_week'] = df.index.dayofweek
    
    return df_feat

# Prediction function
def make_prediction(df, models, scaler):
    cnn_model, lstm_model, xgb_model, meta_model, _ = models
    
    # Create sequences for CNN/LSTM (last 128 hours)
    features = create_features(df)
    scaled_features = scaler.transform(features.fillna(0))
    
    seq_len = 128
    if len(scaled_features) >= seq_len:
        sequence = scaled_features[-seq_len:].reshape(1, seq_len, -1)
        
        # Get individual model predictions
        cnn_pred = cnn_model.predict(sequence, verbose=0)[0][0]
        lstm_pred = lstm_model.predict(sequence, verbose=0)[0][0]
        xgb_pred = xgb_model.predict(scaled_features[-1:].reshape(1, -1))[0]
        
        # Meta features
        meta_features = np.array([[
            cnn_pred, lstm_pred, xgb_pred,
            np.mean([cnn_pred, lstm_pred, xgb_pred]),
            np.std([cnn_pred, lstm_pred, xgb_pred]),
            np.max([cnn_pred, lstm_pred, xgb_pred]),
            np.min([cnn_pred, lstm_pred, xgb_pred])
        ]])
        
        # Final prediction
        final_pred = meta_model.predict(meta_features)[0]
        
        return final_pred, cnn_pred, lstm_pred, xgb_pred
    
    return None, None, None, None

# Title
st.title("SOLARIS: ML-Based Hourly Stock Prediction")
st.markdown("#### Ensemble Deep Learning for Short-Term Market Prediction")
st.markdown("*Using CNN + LSTM + XGBoost with Meta-Learning*")

# Disclaimer
st.warning("⚠️ This is an educational project. Do not use for actual trading decisions.")

# Sidebar
with st.sidebar:
    st.header("Input Parameters")
    ticker = st.text_input("Enter Stock Ticker:", "AAPL").upper()
    lookback_days = st.number_input("Historical Data (90-729 days):", 
                                   min_value=90, max_value=729, value=180)
    
    st.markdown("---")
    st.markdown("### Model Architecture")
    st.image("model_architecture.png")  # You'll need to create this
    
# Main content
col1, col2 = st.columns([2,1])

with col1:
    st.subheader(f"Live Price Chart: {ticker}")
    
    # Get data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval='1h')
        
        if len(df) > 0:
            # Plot candlestick
            fig = go.Figure(data=[go.Candlestick(x=df.index,
                                                open=df['Open'],
                                                high=df['High'],
                                                low=df['Low'],
                                                close=df['Close'])])
            
            fig.update_layout(
                title=f"{ticker} Price Action",
                yaxis_title="Price",
                xaxis_title="Date"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Make prediction
            if st.button("🚀 Predict Next Hour"):
                with st.spinner("Running prediction models..."):
                    models = load_models()
                    pred, cnn_pred, lstm_pred, xgb_pred = make_prediction(df, models)
                    
                    if pred is not None:
                        current_price = df['Close'].iloc[-1]
                        pred_price = current_price * (1 + pred)
                        pred_return = (pred_price/current_price - 1) * 100
                        
                        # Display results
                        st.success(f"Prediction Complete!")
                        
                        metrics_cols = st.columns(3)
                        with metrics_cols[0]:
                            st.metric("Current Price", f"${current_price:.2f}")
                        with metrics_cols[1]:
                            st.metric("Predicted Price", f"${pred_price:.2f}")
                        with metrics_cols[2]:
                            st.metric("Predicted Return", f"{pred_return:.2f}%")
                        
                        # Model votes
                        st.markdown("### Model Votes")
                        votes_cols = st.columns(3)
                        for col, (model, pred) in zip(votes_cols, [
                            ("CNN", cnn_pred > 0),
                            ("LSTM", lstm_pred > 0),
                            ("XGBoost", xgb_pred > 0)
                        ]):
                            with col:
                                st.markdown(f"**{model}**: {'🔺' if pred else '🔻'}")
                        
                        # Trading suggestion
                        st.markdown("### Trading Suggestion")
                        if abs(pred_return) < 0.2:
                            st.info("📊 SIGNAL: HOLD - Predicted move too small")
                        elif pred_return > 0.2:
                            st.success("📈 SIGNAL: BUY - Significant upward move predicted")
                        else:
                            st.error("📉 SIGNAL: SELL - Significant downward move predicted")
                        
                        # Log prediction
                        pred_log = pd.DataFrame({
                            'Timestamp': [datetime.now()],
                            'Ticker': [ticker],
                            'Current_Price': [current_price],
                            'Predicted_Price': [pred_price],
                            'Predicted_Return': [pred_return],
                            'CNN_Vote': [cnn_pred > 0],
                            'LSTM_Vote': [lstm_pred > 0],
                            'XGB_Vote': [xgb_pred > 0]
                        })
                        
                        if 'predictions_log' not in st.session_state:
                            st.session_state.predictions_log = pred_log
                        else:
                            st.session_state.predictions_log = pd.concat([
                                st.session_state.predictions_log, pred_log
                            ])
                        
        else:
            st.error("No data found for this ticker")
            
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")

# Prediction Log
st.markdown("---")
st.header("Prediction History")

if 'predictions_log' in st.session_state:
    st.dataframe(st.session_state.predictions_log, use_container_width=True)
else:
    st.info("No predictions made yet")

# Performance Metrics
if 'predictions_log' in st.session_state and len(st.session_state.predictions_log) > 0:
    st.markdown("### Performance Metrics")
    
    # Calculate metrics (you'll need to implement actual returns checking)
    metrics_cols = st.columns(4)
    with metrics_cols[0]:
        st.metric("Total Predictions", len(st.session_state.predictions_log))
    with metrics_cols[1]:
        st.metric("Avg Predicted Return", 
                 f"{st.session_state.predictions_log['Predicted_Return'].mean():.2f}%")
    with metrics_cols[2]:
        st.metric("Buy Signals", 
                 len(st.session_state.predictions_log[
                     st.session_state.predictions_log['Predicted_Return'] > 0.2
                 ]))
    with metrics_cols[3]:
        st.metric("Sell Signals", 
                 len(st.session_state.predictions_log[
                     st.session_state.predictions_log['Predicted_Return'] < -0.2
                 ]))

# Market Overview
st.markdown("---")
st.header("Market Overview")
indices = ['SPY', 'QQQ', 'DIA']
end = datetime.now()
start = end - timedelta(days=1)

index_cols = st.columns(len(indices))
for idx, symbol in enumerate(indices):
    try:
        data = yf.download(symbol, start=start, end=end, interval='1h')
        if len(data) > 0:
            current = data['Close'].iloc[-1]
            prev = data['Close'].iloc[0]
            change = (current/prev - 1) * 100
            with index_cols[idx]:
                st.metric(symbol, f"${current:.2f}", f"{change:.2f}%")
    except:
        with index_cols[idx]:
            st.metric(symbol, "N/A", "N/A")
