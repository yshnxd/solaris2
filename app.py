import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import ta
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import os
import json
import math
import time
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="SOLARIS: AI Stock Prediction",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Global Styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Animated Background */
    body {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c, #4facfe, #00f2fe);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        font-family: 'Inter', sans-serif;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glassmorphism Effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Main Header */
    .main-header {
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 20px rgba(102, 126, 234, 0.5)); }
        to { filter: drop-shadow(0 0 30px rgba(118, 75, 162, 0.8)); }
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* Prediction Cards */
    .prediction-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .prediction-card:hover::before {
        left: 100%;
    }
    
    .prediction-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
        background: rgba(255, 255, 255, 0.15);
    }
    
    .metric-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        border-radius: 15px 15px 0 0;
    }
    
    /* Color Classes */
    .positive {
        color: #00ff88;
        font-weight: 600;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
    }
    
    .negative {
        color: #ff6b6b;
        font-weight: 600;
        text-shadow: 0 0 10px rgba(255, 107, 107, 0.5);
    }
    
    /* Progress Bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(45deg, #764ba2, #667eea);
    }
    
    /* Text Input */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        color: white;
        backdrop-filter: blur(10px);
    }
    
    .stTextInput > div > div > input:focus {
        border: 1px solid rgba(102, 126, 234, 0.8);
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    /* Slider */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Data Tables */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        overflow: hidden;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: rgba(0, 255, 136, 0.1);
        border: 1px solid rgba(0, 255, 136, 0.3);
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    
    .stError {
        background: rgba(255, 107, 107, 0.1);
        border: 1px solid rgba(255, 107, 107, 0.3);
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    
    .stWarning {
        background: rgba(255, 193, 7, 0.1);
        border: 1px solid rgba(255, 193, 7, 0.3);
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    
    .stInfo {
        background: rgba(52, 144, 220, 0.1);
        border: 1px solid rgba(52, 144, 220, 0.3);
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    
    /* Floating Elements */
    .floating {
        animation: floating 3s ease-in-out infinite;
    }
    
    @keyframes floating {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    /* Pulse Animation */
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Neon Glow */
    .neon-glow {
        text-shadow: 0 0 5px currentColor, 0 0 10px currentColor, 0 0 15px currentColor;
    }
    
    /* Loading Spinner */
    .spinner {
        border: 4px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top: 4px solid #667eea;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Custom Sections */
    .section-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        margin: 3rem 0 2rem 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        position: relative;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 2px;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        .sub-header {
            font-size: 1.2rem;
        }
        .prediction-card {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Hero Section with Floating Elements
st.markdown("""
<div class="floating">
    <h1 class="main-header">🚀 SOLARIS</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pulse">
    <h2 class="sub-header">✨ Next-Gen AI Stock Prediction Engine ✨</h2>
</div>
""", unsafe_allow_html=True)

# Enhanced Description Card
st.markdown("""
<div class="glass-card" style="text-align: center; margin-bottom: 2rem;">
    <div style="font-size: 1.2rem; color: rgba(255, 255, 255, 0.9); line-height: 1.6;">
        🧠 <strong>Advanced Ensemble Learning:</strong> CNN + LSTM + XGBoost + Meta Learner<br>
        📊 <strong>Real-time Predictions:</strong> Hourly stock price forecasting<br>
        🎯 <strong>High Accuracy:</strong> Multi-model consensus for reliable signals<br>
        ⚡ <strong>Lightning Fast:</strong> Instant predictions powered by cutting-edge AI<br><br>
        <em style="color: rgba(255, 255, 255, 0.7); font-size: 0.9rem;">
            ⚠️ Educational & Research Tool • Not Financial Advice
        </em>
    </div>
</div>
""", unsafe_allow_html=True)

PREDICTION_HISTORY_CSV = "prediction_history.csv"
show_cols = ["timestamp", "ticker", "current_price", "predicted_price", "target_time", "actual_price", "error_pct", "error_abs", "confidence", "signal"]

def ensure_csv_has_header(csv_path, columns):
    if not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0:
        pd.DataFrame(columns=columns).to_csv(csv_path, index=False)

def compute_signal(predicted, actual, buy_threshold_pct=0.25, sell_threshold_pct=0.25):
    try:
        predicted = float(predicted)
        actual = float(actual)
    except Exception:
        return "HOLD"
    if actual == 0 or math.isnan(actual):
        return "HOLD"
    pct = (predicted - actual) / actual * 100.0
    if pct >= buy_threshold_pct:
        return "BUY"
    if pct <= -sell_threshold_pct:
        return "SELL"
    return "HOLD"

def batch_evaluate_predictions_csv(csv_path, rate_limit_sec=0.5, hour_fetch_period="730d"):
    ensure_csv_has_header(csv_path, show_cols)
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=show_cols)
    for col in ['timestamp', 'target_time']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
    for col in ['actual_price', 'error_pct', 'error_abs', 'signal']:
        if col not in df.columns:
            df[col] = np.nan
    df['signal'] = df['signal'].fillna('HOLD')
    ticker_groups = {}
    for idx, row in df.iterrows():
        t = str(row.get('ticker','')).strip()
        if t=="" or pd.isna(row.get('target_time')):
            continue
        ticker_groups.setdefault(t, []).append(idx)
    for j, (ticker, indices) in enumerate(ticker_groups.items(), start=1):
        yf_ticker = ticker
        try:
            tk = yf.Ticker(yf_ticker)
            hist_hour = tk.history(period=hour_fetch_period, interval='1h', actions=False, auto_adjust=False)
            if isinstance(hist_hour.index, pd.DatetimeIndex):
                if hist_hour.index.tz is None:
                    hist_hour.index = hist_hour.index.tz_localize('UTC')
                else:
                    hist_hour.index = hist_hour.index.tz_convert('UTC')
        except Exception:
            hist_hour = None
        try:
            hist_day = tk.history(period='365d', interval='1d', actions=False, auto_adjust=False)
            if isinstance(hist_day.index, pd.DatetimeIndex):
                if hist_day.index.tz is None:
                    hist_day.index = hist_day.index.tz_localize('UTC')
                else:
                    hist_day.index = hist_day.index.tz_convert('UTC')
        except Exception:
            hist_day = None
        for i in indices:
            t_utc = df.at[i, 'target_time']
            if pd.isna(t_utc):
                continue
            if not pd.isna(df.at[i, 'actual_price']):
                continue
            filled = False
            if hist_hour is not None and len(hist_hour) > 0:
                target_hour = t_utc.floor('H')
                if target_hour in hist_hour.index:
                    close_val = float(hist_hour.loc[target_hour]['Close'])
                    df.at[i, 'actual_price'] = close_val
                    filled = True
                else:
                    diffs = (hist_hour.index - t_utc).total_seconds()
                    best_idx = np.abs(diffs).argmin()
                    close_val = float(hist_hour.iloc[best_idx]['Close'])
                    df.at[i, 'actual_price'] = close_val
                    filled = True
                if filled:
                    try:
                        predicted = float(df.at[i, 'predicted_price'])
                        if not math.isnan(predicted) and close_val != 0:
                            df.at[i, 'error_pct'] = (close_val - predicted) / close_val * 100.0
                            df.at[i, 'error_abs'] = abs(close_val - predicted)
                    except Exception:
                        pass
                    df.at[i, 'signal'] = compute_signal(df.at[i, 'predicted_price'], close_val)
            if not filled:
                if hist_day is not None and len(hist_day) > 0:
                    target_date = t_utc.normalize()
                    candidate = hist_day.loc[hist_day.index <= target_date]
                    if candidate is None or len(candidate) == 0:
                        candidate = hist_day
                    close_val = float(candidate.iloc[-1]['Close'])
                    df.at[i, 'actual_price'] = close_val
                    try:
                        predicted = float(df.at[i, 'predicted_price'])
                        if not math.isnan(predicted) and close_val != 0:
                            df.at[i, 'error_pct'] = (close_val - predicted) / close_val * 100.0
                            df.at[i, 'error_abs'] = abs(close_val - predicted)
                    except Exception:
                        pass
                    df.at[i, 'signal'] = compute_signal(df.at[i, 'predicted_price'], close_val)
        time.sleep(rate_limit_sec)
    df.to_csv(csv_path, index=False)
    return df

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

original_cross_assets = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "NVDA", "JPM", "JNJ", "XOM", "CAT", "BA", "META"]
dynamic_cross_assets = original_cross_assets.copy()
expected_columns = [
    'ret_1h', 'ret_3h', 'ret_6h', 'ret_12h', 'ret_24h', 'vol_6h', 'vol_12h', 'vol_24h',
    'rsi_14', 'macd', 'macd_signal', 'sma_5', 'ema_5', 'sma_10', 'ema_10', 'sma_20', 'ema_20',
    'vol_change_1h', 'vol_ma_24h', 'AAPL_ret_1h', 'MSFT_ret_1h', 'AMZN_ret_1h', 'GOOGL_ret_1h',
    'TSLA_ret_1h', 'NVDA_ret_1h', 'JPM_ret_1h', 'JNJ_ret_1h', 'XOM_ret_1h', 'CAT_ret_1h',
    'BA_ret_1h', 'META_ret_1h', 'hour', 'day_of_week', 'price'
]
SEQ_LEN = 128

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

def create_features_for_app(selected_ticker, ticker_data, cross_data):
    price_series = ticker_data['Close']
    feat_tmp = pd.DataFrame(index=price_series.index)
    for lag in [1, 3, 6, 12, 24]:
        feat_tmp[f"ret_{lag}h"] = price_series.pct_change(lag)
    for window in [6, 12, 24]:
        feat_tmp[f"vol_{window}h"] = price_series.pct_change().rolling(window).std()
    try:
        feat_tmp["rsi_14"] = ta.momentum.RSIIndicator(price_series, window=14).rsi()
    except Exception:
        feat_tmp["rsi_14"] = 0
    try:
        macd = ta.trend.MACD(price_series)
        feat_tmp["macd"] = macd.macd()
        feat_tmp["macd_signal"] = macd.macd_signal()
    except Exception:
        feat_tmp["macd"] = 0
        feat_tmp["macd_signal"] = 0
    for w in [5, 10, 20]:
        feat_tmp[f"sma_{w}"] = price_series.rolling(w).mean()
        feat_tmp[f"ema_{w}"] = price_series.ewm(span=w, adjust=False).mean()
    if "Volume" in ticker_data.columns:
        vol_series = ticker_data['Volume'].ffill()
        feat_tmp["vol_change_1h"] = vol_series.pct_change()
        feat_tmp["vol_ma_24h"] = vol_series.rolling(24).mean()
    else:
        feat_tmp["vol_change_1h"] = 0
        feat_tmp["vol_ma_24h"] = 0
    for asset in original_cross_assets:
        feat_tmp[f"{asset}_ret_1h"] = cross_data.get(asset, pd.Series(0, index=feat_tmp.index)).pct_change().fillna(0)
    feat_tmp["hour"] = feat_tmp.index.hour
    feat_tmp["day_of_week"] = feat_tmp.index.dayofweek
    feat_tmp["price"] = price_series
    for col in ["rsi_14", "macd", "macd_signal", "vol_change_1h", "vol_ma_24h"]:
        feat_tmp[col] = feat_tmp[col].fillna(0)
    feat_tmp = feat_tmp.dropna(subset=["price"])
    for col in expected_columns:
        if col not in feat_tmp.columns:
            feat_tmp[col] = 0
    feat_tmp = feat_tmp[expected_columns]
    return feat_tmp

def predict_with_models(features, current_price, scaler, cnn_model, lstm_model, xgb_model, meta_model):
    if scaler is None or cnn_model is None or lstm_model is None or xgb_model is None or meta_model is None:
        return None, None, None, None
    features = features[expected_columns]
    features = features.replace([np.inf, -np.inf], 0).fillna(0)
    if len(features) < SEQ_LEN:
        return None, None, None, None
    features_recent = features.iloc[-SEQ_LEN:]
    features_tabular = features.iloc[-1:]
    features_recent = features_recent.replace([np.inf, -np.inf], 0).fillna(0)
    features_tabular = features_tabular.replace([np.inf, -np.inf], 0).fillna(0)
    X_seq = scaler.transform(features_recent)
    X_tab = scaler.transform(features_tabular)
    X_seq = X_seq.reshape(1, SEQ_LEN, -1)
    X_tab = X_tab.reshape(1, -1)
    try:
        pred_cnn = cnn_model.predict(X_seq, verbose=0)
        pred_cnn = float(pred_cnn[0][0])
    except Exception as e:
        pred_cnn = 0.0
    try:
        pred_lstm = lstm_model.predict(X_seq, verbose=0)
        pred_lstm = float(pred_lstm[0][0])
    except Exception as e:
        pred_lstm = 0.0
    try:
        pred_xgb = xgb_model.predict(X_tab)
        pred_xgb = float(pred_xgb[0])
    except Exception as e:
        pred_xgb = 0.0
    meta_feats = np.array([pred_cnn, pred_lstm, pred_xgb])
    meta_stats = np.array([meta_feats.mean(), meta_feats.std(), meta_feats.max(), meta_feats.min()])
    meta_input = np.concatenate([meta_feats, meta_stats]).reshape(1, -1)
    try:
        pred_meta = meta_model.predict(meta_input)
        pred_meta = float(pred_meta[0])
    except Exception as e:
        pred_meta = np.mean([pred_cnn, pred_lstm, pred_xgb])
    predicted_price = current_price * (1 + pred_meta)
    pred_change_pct = pred_meta * 100
    votes = {
        "CNN": "UP" if pred_cnn > 0 else "DOWN",
        "LSTM": "UP" if pred_lstm > 0 else "DOWN",
        "XGBoost": "UP" if pred_xgb > 0 else "DOWN"
    }
    agreement = [pred_cnn, pred_lstm, pred_xgb]
    up_count = sum([1 for v in agreement if v > 0])
    down_count = sum([1 for v in agreement if v < 0])
    confidence = (max(up_count, down_count) / 3) * 100
    return predicted_price, pred_change_pct, votes, confidence

# Enhanced Sidebar
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid rgba(255, 255, 255, 0.2); margin-bottom: 1rem;">
    <h3 style="color: rgba(255, 255, 255, 0.9); margin: 0; font-weight: 600;">⚙️ Control Panel</h3>
</div>
""", unsafe_allow_html=True)

# Stock Selection with Enhanced Styling
st.sidebar.markdown("### 📈 Stock Selection")
selected_ticker = st.sidebar.text_input("Stock Ticker Symbol", "AAPL", help="Enter a valid stock ticker symbol (e.g., AAPL, MSFT, TSLA)").upper()

# History Slider with Enhanced Styling
st.sidebar.markdown("### 📊 Data Range")
days_history = st.sidebar.slider(
    "Historical Data Period", 
    min_value=30, 
    max_value=729, 
    value=90,
    help="Number of days of historical data to analyze"
)

# Action Buttons with Enhanced Styling
st.sidebar.markdown("### 🚀 Actions")
run_prediction = st.sidebar.button("🎯 Generate Prediction", help="Run AI prediction for selected stock")
evaluate_results = st.sidebar.button("📋 Evaluate History", help="Fill in actual prices for previous predictions")
run_watchlist = st.sidebar.button("📊 Watchlist Analysis", help="Run predictions for all watchlist stocks")

# Chart Configuration
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <h4 style="color: rgba(255, 255, 255, 0.9); margin: 0; font-weight: 600;">📈 Chart Configuration</h4>
</div>
""", unsafe_allow_html=True)

# Enhanced Indicator Controls
st.sidebar.markdown("#### Price Overlays")
overlay_options = st.sidebar.multiselect(
    "Select Moving Averages",
    ["SMA 5", "SMA 10", "SMA 20", "EMA 5", "EMA 10", "EMA 20"],
    default=["SMA 20"],
    help="Choose which moving averages to display on the chart"
)

st.sidebar.markdown("#### Technical Indicators")
show_rsi = st.sidebar.checkbox("📊 RSI (14)", value=False, help="Relative Strength Index")
show_macd = st.sidebar.checkbox("📈 MACD", value=False, help="Moving Average Convergence Divergence")

main_ticker_data = fetch_stock_data(selected_ticker, period=f"{days_history}d", interval="60m")

if main_ticker_data is None or main_ticker_data.empty or len(main_ticker_data) < 128:
    st.error(f"No or insufficient data found for ticker: {selected_ticker}. Please check the symbol or try another stock.")
else:
    latest_data = main_ticker_data.iloc[-1]
    current_price = latest_data['Close']
    current_time = latest_data.name
    
    # Enhanced Metrics Display
    st.markdown('<h2 class="section-header">📊 Real-Time Market Data</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_price_formatted = f"${current_price:.2f}"
        st.markdown(f"""
        <div class="glass-card metric-card">
            <div style="text-align: center;">
                <div style="font-size: 2.5rem; font-weight: 900; color: #00ff88; margin-bottom: 0.5rem;" class="neon-glow">
                    {current_price_formatted}
                </div>
                <div style="font-size: 1.1rem; color: rgba(255, 255, 255, 0.8); font-weight: 600;">
                    💰 Current Price
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        prev_close = main_ticker_data.iloc[-2]['Close'] if len(main_ticker_data) > 1 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100
        change_color = "#00ff88" if change >= 0 else "#ff6b6b"
        change_icon = "📈" if change >= 0 else "📉"
        change_formatted = f"{change:+.2f}"
        change_pct_formatted = f"{change_pct:+.2f}%"
        
        st.markdown(f"""
        <div class="glass-card metric-card">
            <div style="text-align: center;">
                <div style="font-size: 2rem; font-weight: 900; color: {change_color}; margin-bottom: 0.5rem;" class="neon-glow">
                    {change_icon} {change_formatted}
                </div>
                <div style="font-size: 1.3rem; color: {change_color}; font-weight: 700; margin-bottom: 0.5rem;">
                    {change_pct_formatted}
                </div>
                <div style="font-size: 1.1rem; color: rgba(255, 255, 255, 0.8); font-weight: 600;">
                    ⚡ Price Change
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="glass-card metric-card">
            <div style="text-align: center;">
                <div style="font-size: 1.8rem; font-weight: 900; color: #667eea; margin-bottom: 0.5rem;" class="neon-glow">
                    🕐 {current_time.strftime("%H:%M")}
                </div>
                <div style="font-size: 1rem; color: rgba(255, 255, 255, 0.8); margin-bottom: 0.5rem;">
                    {current_time.strftime("%Y-%m-%d")}
                </div>
                <div style="font-size: 1.1rem; color: rgba(255, 255, 255, 0.8); font-weight: 600;">
                    📅 Last Updated
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    # Compute indicators for plotting
    close_series = main_ticker_data['Close']
    sma_5 = close_series.rolling(5).mean() if len(close_series) >= 5 else None
    sma_10 = close_series.rolling(10).mean() if len(close_series) >= 10 else None
    sma_20 = close_series.rolling(20).mean() if len(close_series) >= 20 else None
    ema_5 = close_series.ewm(span=5, adjust=False).mean() if len(close_series) >= 5 else None
    ema_10 = close_series.ewm(span=10, adjust=False).mean() if len(close_series) >= 10 else None
    ema_20 = close_series.ewm(span=20, adjust=False).mean() if len(close_series) >= 20 else None
    try:
        rsi_14 = ta.momentum.RSIIndicator(close_series, window=14).rsi() if show_rsi else None
    except Exception:
        rsi_14 = None
    try:
        macd_obj = ta.trend.MACD(close_series) if show_macd else None
        macd_line = macd_obj.macd() if macd_obj is not None else None
        macd_signal = macd_obj.macd_signal() if macd_obj is not None else None
        macd_hist = macd_obj.macd_diff() if macd_obj is not None else None
    except Exception:
        macd_line = macd_signal = macd_hist = None

    # Dynamic subplot layout: Price, optional RSI, optional MACD, Volume
    extra_rows = (1 if show_rsi else 0) + (1 if show_macd else 0)
    total_rows = 2 + extra_rows
    subplot_titles = [f'{selected_ticker} Price']
    if show_rsi:
        subplot_titles.append('RSI (14)')
    if show_macd:
        subplot_titles.append('MACD')
    subplot_titles.append('Volume')
    # Row widths: put more height on price; smaller on indicators and volume
    row_width = []
    # Build from bottom to top as Plotly expects
    # Start with Volume
    row_width.append(0.2)
    # Add MACD if used
    if show_macd:
        row_width.append(0.25)
    # Add RSI if used
    if show_rsi:
        row_width.append(0.25)
    # Finally price
    row_width.append(0.8)
    fig = make_subplots(rows=total_rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.09,
                        subplot_titles=tuple(subplot_titles),
                        row_width=row_width)
    # Row indices
    price_row = 1
    vol_row = total_rows
    rsi_row = 2 if show_rsi else None
    macd_row = (3 if show_rsi else 2) if show_macd else None

    # Price + overlays
    fig.add_trace(go.Candlestick(
        x=main_ticker_data.index,
                                 open=main_ticker_data['Open'],
                                 high=main_ticker_data['High'],
                                 low=main_ticker_data['Low'],
                                 close=main_ticker_data['Close'],
        name="Price"
    ), row=price_row, col=1)
    overlay_map = {
        "SMA 5": (sma_5, "SMA 5", "#7f7f7f", "dash", "SMA"),
        "SMA 10": (sma_10, "SMA 10", "#b0b0b0", "dash", "SMA"),
        "SMA 20": (sma_20, "SMA 20", "#c7c7c7", "dash", "SMA"),
        "EMA 5": (ema_5, "EMA 5", "#ffbb78", "dot", "EMA"),
        "EMA 10": (ema_10, "EMA 10", "#aec7e8", "dot", "EMA"),
        "EMA 20": (ema_20, "EMA 20", "#c5b0d5", "dot", "EMA"),
    }
    for key in overlay_options:
        series, label, color, dash, group = overlay_map.get(key, (None, None, None, None, None))
        if series is not None:
            fig.add_trace(go.Scatter(
                x=series.index,
                y=series,
                name=label,
                mode='lines',
                line=dict(color=color, width=1.2, dash=dash),
                legendgroup=group,
                hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>"+label+": %{y:.2f}<extra></extra>"
            ), row=price_row, col=1)

    # RSI subplot
    if show_rsi and rsi_row is not None and rsi_14 is not None:
        # Shaded zones: green band 30-70, light red outside (0-30 and 70-100)
        fig.add_hrect(y0=30, y1=70, fillcolor="rgba(46, 204, 113, 0.08)", line_width=0, row=rsi_row, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(231, 76, 60, 0.06)", line_width=0, row=rsi_row, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(231, 76, 60, 0.06)", line_width=0, row=rsi_row, col=1)
        fig.add_trace(go.Scatter(
            x=rsi_14.index, y=rsi_14, name='RSI (14)', mode='lines',
            line=dict(color='#1f77b4', width=1.4),
            hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>RSI: %{y:.2f}<extra></extra>"
        ), row=rsi_row, col=1)
        # 30/70 guide lines
        fig.add_hline(y=70, line_dash="dot", line_color="#e74c3c", row=rsi_row, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#2ecc71", row=rsi_row, col=1)

    # MACD subplot
    if show_macd and macd_row is not None and macd_line is not None:
        if macd_hist is not None:
            fig.add_trace(go.Bar(
                x=macd_hist.index,
                y=macd_hist,
                name='MACD Hist',
                marker_color=['#2ecc71' if v >= 0 else '#e74c3c' for v in macd_hist.fillna(0)],
                opacity=0.5,
                hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Hist: %{y:.4f}<extra></extra>"
            ), row=macd_row, col=1)
        fig.add_trace(go.Scatter(
            x=macd_line.index, y=macd_line, name='MACD', mode='lines',
            line=dict(color='#1f77b4', width=1.4),
            hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>MACD: %{y:.4f}<extra></extra>"
        ), row=macd_row, col=1)
        if macd_signal is not None:
            fig.add_trace(go.Scatter(
                x=macd_signal.index, y=macd_signal, name='Signal', mode='lines',
                line=dict(color='#ff7f0e', width=1.2, dash='dot'),
                hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Signal: %{y:.4f}<extra></extra>"
            ), row=macd_row, col=1)

    # Volume subplot (always)
    colors = ['rgba(231,76,60,0.45)' if row['Open'] > row['Close'] else 'rgba(46,204,113,0.45)'
              for _, row in main_ticker_data.iterrows()]
    fig.add_trace(go.Bar(x=main_ticker_data.index,
                         y=main_ticker_data['Volume'],
                         marker_color=colors,
                         name="Volume",
                         hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Vol: %{y:.0f}<extra>Volume</extra>"), row=vol_row, col=1)
    fig.add_vline(x=current_time, line_dash="dash", line_color="rgba(255,255,255,0.5)", row=price_row, col=1)
    # Enhanced Chart Theme
    fig.update_layout(
        height=760 if extra_rows else 620,
        showlegend=True,
        legend=dict(
            orientation='h', 
            yanchor='bottom', 
            y=1.02, 
            xanchor='left', 
            x=0, 
            groupclick='toggleitem',
            bgcolor='rgba(255, 255, 255, 0.1)',
            bordercolor='rgba(255, 255, 255, 0.2)',
            borderwidth=1,
            font=dict(color='rgba(255, 255, 255, 0.9)')
        ),
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor='rgba(255, 255, 255, 0.05)',
        paper_bgcolor='rgba(255, 255, 255, 0.05)',
        font=dict(color='rgba(255, 255, 255, 0.9)', family='Inter'),
        title=dict(
            text=f"{selected_ticker} Price Chart",
            font=dict(size=24, color='rgba(255, 255, 255, 0.9)'),
            x=0.5
        )
    )
    
    # Enhanced axis styling
    fig.update_xaxes(
        gridcolor='rgba(255, 255, 255, 0.1)',
        color='rgba(255, 255, 255, 0.8)',
        showgrid=True
    )
    fig.update_yaxes(
        gridcolor='rgba(255, 255, 255, 0.1)',
        color='rgba(255, 255, 255, 0.8)',
        showgrid=True
    )
    # Fix RSI range for clarity
    if show_rsi and rsi_row is not None:
        fig.update_yaxes(range=[0, 100], row=rsi_row, col=1)
    st.plotly_chart(fig, use_container_width=True)

    if cnn_model and lstm_model and xgb_model and meta_model and scaler:
        cross_data = fetch_cross_assets(main_ticker_data.index, days_history, selected_ticker)
        run_watchlist_triggered = run_watchlist
        if run_prediction:
            with st.spinner('Generating prediction...'):
                features = create_features_for_app(selected_ticker, main_ticker_data, cross_data)
                if len(features) < 128:
                    st.error("Not enough historical data to generate prediction. Need at least 128 hours of data.")
                else:
                    predicted_price, pred_change_pct, votes, confidence = predict_with_models(
                        features, current_price, scaler, cnn_model, lstm_model, xgb_model, meta_model
                    )
                    if predicted_price is not None:
                        vote_display = {
                            "CNN": "UP" if votes["CNN"] == "UP" else "DOWN",
                            "LSTM": "UP" if votes["LSTM"] == "UP" else "DOWN",
                            "XGBoost": "UP" if votes["XGBoost"] == "UP" else "DOWN"
                        }
                        st.markdown('<h2 class="section-header">🎯 AI Prediction Results</h2>', unsafe_allow_html=True)
                        
                        # Prediction Cards with Enhanced Styling
                        pred_col1, pred_col2, pred_col3 = st.columns(3)
                        
                        with pred_col1:
                            pred_color = "#00ff88" if pred_change_pct > 0 else "#ff6b6b"
                            pred_icon = "🚀" if pred_change_pct > 1 else "📈" if pred_change_pct > 0 else "📉" if pred_change_pct < -1 else "📊"
                            predicted_price_formatted = f"${predicted_price:.2f}"
                            pred_change_pct_formatted = f"{pred_change_pct:+.2f}%"
                            
                            st.markdown(f"""
                            <div class="glass-card prediction-card pulse">
                                <div style="text-align: center;">
                                    <div style="font-size: 3rem; margin-bottom: 1rem;">{pred_icon}</div>
                                    <div style="font-size: 2.5rem; font-weight: 900; color: {pred_color}; margin-bottom: 0.5rem;" class="neon-glow">
                                        {predicted_price_formatted}
                                    </div>
                                    <div style="font-size: 1.5rem; color: {pred_color}; font-weight: 700; margin-bottom: 1rem;">
                                        {pred_change_pct_formatted}
                                    </div>
                                    <div style="font-size: 1.2rem; color: rgba(255, 255, 255, 0.9); font-weight: 600;">
                                        🎯 Predicted Price (1 Hour)
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        with pred_col2:
                            # Model Consensus Card
                            st.markdown("""
                            <div class="glass-card prediction-card">
                                <div style="text-align: center;">
                                    <div style="font-size: 2rem; margin-bottom: 1rem;">🧠</div>
                                    <div style="font-size: 1.3rem; color: rgba(255, 255, 255, 0.9); font-weight: 600; margin-bottom: 1rem;">
                                        AI Model Consensus
                                    </div>
                            """, unsafe_allow_html=True)
                            
                            # Model votes with enhanced styling
                            for model, vote in vote_display.items():
                                vote_color = "#00ff88" if vote == "UP" else "#ff6b6b"
                                vote_icon = "🟢" if vote == "UP" else "🔴"
                                st.markdown(f"""
                                <div style="display: flex; justify-content: space-between; align-items: center; margin: 0.5rem 0; padding: 0.5rem; background: rgba(255, 255, 255, 0.1); border-radius: 10px;">
                                    <span style="color: rgba(255, 255, 255, 0.9); font-weight: 600;">{model}</span>
                                    <span style="color: {vote_color}; font-weight: 700;">{vote_icon} {vote}</span>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Confidence with enhanced progress bar
                            conf_color = "#00ff88" if confidence > 70 else "#ffa500" if confidence > 50 else "#ff6b6b"
                            st.markdown(f"""
                                <div style="margin-top: 1rem;">
                                    <div style="font-size: 1.1rem; color: rgba(255, 255, 255, 0.9); font-weight: 600; margin-bottom: 0.5rem;">
                                        🎯 Confidence Level
                                    </div>
                                    <div style="background: rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 0.5rem; margin-bottom: 0.5rem;">
                                        <div style="background: {conf_color}; height: 20px; border-radius: 10px; width: {confidence}%; transition: width 0.3s ease;"></div>
                                    </div>
                                    <div style="font-size: 1.5rem; color: {conf_color}; font-weight: 900; text-align: center;">
                                        {confidence:.1f}%
                                    </div>
                                </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        with pred_col3:
                            # Trading Signal Card
                            if pred_change_pct > 1.0 and confidence > 70:
                                signal_color = "#00ff88"
                                signal_bg = "rgba(0, 255, 136, 0.2)"
                                signal_icon = "🚀"
                                signal_text = "STRONG BUY"
                                signal_desc = f"High confidence prediction of {pred_change_pct:.2f}% increase"
                            elif pred_change_pct > 0.5:
                                signal_color = "#4CAF50"
                                signal_bg = "rgba(76, 175, 80, 0.2)"
                                signal_icon = "📈"
                                signal_text = "BUY"
                                signal_desc = f"Predicted {pred_change_pct:.2f}% increase"
                            elif pred_change_pct < -1.0 and confidence > 70:
                                signal_color = "#ff6b6b"
                                signal_bg = "rgba(255, 107, 107, 0.2)"
                                signal_icon = "📉"
                                signal_text = "STRONG SELL"
                                signal_desc = f"High confidence prediction of {abs(pred_change_pct):.2f}% decrease"
                            elif pred_change_pct < -0.5:
                                signal_color = "#ff9800"
                                signal_bg = "rgba(255, 152, 0, 0.2)"
                                signal_icon = "📊"
                                signal_text = "SELL"
                                signal_desc = f"Predicted {abs(pred_change_pct):.2f}% decrease"
                            else:
                                signal_color = "#667eea"
                                signal_bg = "rgba(102, 126, 234, 0.2)"
                                signal_icon = "⏸️"
                                signal_text = "HOLD"
                                signal_desc = "No strong directional signal detected"
                            
                            st.markdown(f"""
                            <div class="glass-card prediction-card" style="border: 2px solid {signal_color}; background: {signal_bg};">
                                <div style="text-align: center;">
                                    <div style="font-size: 3rem; margin-bottom: 1rem;">{signal_icon}</div>
                                    <div style="font-size: 1.8rem; color: {signal_color}; font-weight: 900; margin-bottom: 1rem;" class="neon-glow">
                                        {signal_text}
                                    </div>
                                    <div style="font-size: 1.1rem; color: rgba(255, 255, 255, 0.9); line-height: 1.4; margin-bottom: 1rem;">
                                        {signal_desc}
                                    </div>
                                    <div style="font-size: 1rem; color: rgba(255, 255, 255, 0.7); font-weight: 600;">
                                        🎯 Trading Signal
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        target_time = current_time + pd.Timedelta(hours=1)
                        ensure_csv_has_header(PREDICTION_HISTORY_CSV, show_cols)
                        csv_row = {
                            "timestamp": datetime.now(),
                            "ticker": selected_ticker,
                            "current_price": float(current_price),
                            "predicted_price": float(predicted_price),
                            "predicted_change": float(pred_change_pct),
                            "confidence": float(confidence),
                            "signal": "BUY" if pred_change_pct > 0.5 else "SELL" if pred_change_pct < -0.5 else "HOLD",
                            "target_time": target_time,
                            "actual_price": np.nan,
                            "error_pct": np.nan,
                            "error_abs": np.nan,
                        }
                        if os.path.exists(PREDICTION_HISTORY_CSV):
                            df_log = pd.read_csv(PREDICTION_HISTORY_CSV)
                            df_log = pd.concat([df_log, pd.DataFrame([csv_row])], ignore_index=True)
                        else:
                            df_log = pd.DataFrame([csv_row])
                        df_log.to_csv(PREDICTION_HISTORY_CSV, index=False)
                        st.info(f"Prediction logged to {PREDICTION_HISTORY_CSV}. Next hour target_time is {target_time}. Use the 'Evaluate Predictions' button in the sidebar to fill in actual prices for the predictions.")
        if run_watchlist_triggered:
            with st.spinner('Running watchlist predictions...'):
                ensure_csv_has_header(PREDICTION_HISTORY_CSV, show_cols)
                appended = 0
                for tk in original_cross_assets:
                    df_tk = fetch_stock_data(tk, period=f"{days_history}d", interval="60m")
                    if df_tk is None or df_tk.empty or len(df_tk) < SEQ_LEN:
                        continue
                    latest_tk = df_tk.iloc[-1]
                    current_price_tk = latest_tk['Close']
                    current_time_tk = latest_tk.name
                    cross_tk = fetch_cross_assets(df_tk.index, days_history, tk)
                    feats_tk = create_features_for_app(tk, df_tk, cross_tk)
                    if len(feats_tk) < SEQ_LEN:
                        continue
                    pred_price_tk, pred_change_pct_tk, votes_tk, conf_tk = predict_with_models(
                        feats_tk, current_price_tk, scaler, cnn_model, lstm_model, xgb_model, meta_model
                    )
                    if pred_price_tk is None:
                        continue
                    signal_tk = "BUY" if pred_change_pct_tk > 0.5 else "SELL" if pred_change_pct_tk < -0.5 else "HOLD"
                    target_time_tk = current_time_tk + pd.Timedelta(hours=1)
                    csv_row = {
                        "timestamp": datetime.now(),
                        "ticker": tk,
                        "current_price": float(current_price_tk),
                        "predicted_price": float(pred_price_tk),
                        "confidence": float(conf_tk),
                        "signal": signal_tk,
                        "target_time": target_time_tk,
                        "actual_price": np.nan,
                        "error_pct": np.nan,
                        "error_abs": np.nan,
                    }
                    if os.path.exists(PREDICTION_HISTORY_CSV):
                        df_log = pd.read_csv(PREDICTION_HISTORY_CSV)
                        df_log = pd.concat([df_log, pd.DataFrame([csv_row])], ignore_index=True)
                    else:
                        df_log = pd.DataFrame([csv_row])
                    df_log.to_csv(PREDICTION_HISTORY_CSV, index=False)
                    appended += 1
                if appended == 0:
                    st.warning("No watchlist predictions could be generated (insufficient data).")
                else:
                    st.success(f"Appended {appended} watchlist predictions to {PREDICTION_HISTORY_CSV}. See the History Prediction table below.")
        if evaluate_results:
            with st.spinner("Evaluating prediction history and filling actuals..."):
                updated_df = batch_evaluate_predictions_csv(PREDICTION_HISTORY_CSV)
                if updated_df is not None:
                    st.success("Filled actual prices and updated prediction history CSV.")
        ensure_csv_has_header(PREDICTION_HISTORY_CSV, show_cols)
        try:
            log_df = pd.read_csv(PREDICTION_HISTORY_CSV)
        except pd.errors.EmptyDataError:
            log_df = pd.DataFrame(columns=show_cols)
        for col in ["timestamp", "target_time"]:
            if col in log_df.columns:
                log_df[col] = pd.to_datetime(log_df[col], errors="coerce")
        log_df = log_df.sort_values("timestamp", ascending=False)
        for col in show_cols:
            if col not in log_df.columns:
                log_df[col] = np.nan
        st.dataframe(log_df[show_cols], use_container_width=True)
        if 'error_pct' in log_df.columns and log_df['error_pct'].notnull().any():
            eval_rows = log_df[log_df['error_pct'].notnull()]
            avg_abs_error = eval_rows['error_abs'].mean()
            avg_pct_error = eval_rows['error_pct'].mean()
            eval_count = len(eval_rows)
            st.markdown("### Predictions Summary")
            st.metric("Evaluated Predictions", eval_count)
            st.metric("Average Absolute Error", f"${avg_abs_error:.3f}" if eval_count else "N/A")
            st.metric("Average % Error", f"{avg_pct_error:.2f}%" if eval_count else "N/A")
            st.progress(min(1, max(0, 1-avg_pct_error/10)) if eval_count else 0)

# --- HISTORY PREDICTION (SOLARIS) SECTION ---
st.markdown('<h2 class="section-header">📋 Prediction History</h2>', unsafe_allow_html=True)
solaris_csv_path = "solaris-data.csv"
if os.path.exists(solaris_csv_path):
    hist_df = pd.read_csv(solaris_csv_path)
    for col in ["timestamp", "target_time"]:
        if col in hist_df.columns:
            hist_df[col] = pd.to_datetime(hist_df[col], errors="coerce")
    if "evaluated_time" in hist_df.columns:
        hist_df["evaluated_time"] = pd.to_datetime(hist_df["evaluated_time"], errors="coerce")
    hist_df = hist_df.sort_values("timestamp", ascending=False)
    display_cols = ["timestamp", "ticker", "current_price", "predicted_price", "target_time", "actual_price", "error_abs", "error_pct", "confidence", "signal"]
    for col in display_cols:
        if col not in hist_df.columns:
            hist_df[col] = np.nan
    st.dataframe(hist_df[display_cols], use_container_width=True)
    valid_rows = hist_df.dropna(subset=["error_abs", "error_pct"])
    avg_abs_error = valid_rows["error_abs"].mean() if not valid_rows.empty else float("nan")
    avg_pct_error = valid_rows["error_pct"].mean() if not valid_rows.empty else float("nan")
    count = len(valid_rows)
    st.markdown("### Backtest Summary")
    st.metric("Evaluated Predictions", count)
    st.metric("Average Absolute Error", f"${avg_abs_error:.3f}" if count else "N/A")
    st.metric("Average % Error", f"{avg_pct_error:.2f}%" if count else "N/A")
    st.progress(min(1, max(0, 1-avg_pct_error/10)) if count else 0)
else:
    st.warning("No solaris-data.csv file found for history prediction results.")

st.markdown("---")
st.markdown('<h2 class="section-header">🔬 How SOLARIS Works</h2>', unsafe_allow_html=True)
exp_col1, exp_col2 = st.columns(2)
with exp_col1:
    st.markdown("""
    <div class="glass-card" style="margin-bottom: 2rem;">
        <h3 style="color: rgba(255, 255, 255, 0.9); margin-bottom: 1rem; text-align: center;">🧠 Model Architecture</h3>
        <div style="color: rgba(255, 255, 255, 0.8); line-height: 1.6;">
            SOLARIS uses an advanced ensemble of three cutting-edge machine learning models:<br><br>
            <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                <strong>🔍 CNN (Convolutional Neural Network)</strong><br>
                Identifies complex patterns in price movements and market trends
            </div>
            <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                <strong>🔄 LSTM (Long Short-Term Memory)</strong><br>
                Captures sequential dependencies and temporal patterns in time series data
            </div>
            <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                <strong>⚡ XGBoost</strong><br>
                Handles structured features and non-linear relationships with gradient boosting
            </div>
            <div style="text-align: center; margin-top: 1rem; padding: 1rem; background: linear-gradient(45deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2)); border-radius: 10px;">
                <strong>🎯 Meta-Learner Ensemble</strong><br>
                Combines all predictions for maximum accuracy and reliability
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: rgba(255, 255, 255, 0.9); margin-bottom: 1rem; text-align: center;">📊 Top Predictive Features</h3>
        <div style="color: rgba(255, 255, 255, 0.8);">
    """, unsafe_allow_html=True)
    
    feature_importance = {
        "Previous Hour Return": 0.18,
        "RSI (14-period)": 0.15,
        "MACD Signal": 0.12,
        "Volume Change": 0.11,
        "NVDA 1h Return": 0.09,
        "MSFT 1h Return": 0.08,
        "Volatility (24h)": 0.07,
        "SMA (20-period)": 0.06,
        "Hour of Day": 0.05,
        "Day of Week": 0.04
    }
    
    for feature, importance in feature_importance.items():
        st.markdown(f"""
        <div style="margin: 0.5rem 0; padding: 0.5rem; background: rgba(255, 255, 255, 0.1); border-radius: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
                <span style="color: rgba(255, 255, 255, 0.9); font-weight: 600;">{feature}</span>
                <span style="color: #00ff88; font-weight: 700;">{importance:.0%}</span>
            </div>
            <div style="background: rgba(255, 255, 255, 0.1); border-radius: 10px; height: 8px;">
                <div style="background: linear-gradient(90deg, #667eea, #764ba2); height: 8px; border-radius: 10px; width: {importance*100}%; transition: width 0.3s ease;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        </div>
    </div>
    """, unsafe_allow_html=True)
with exp_col2:
    st.markdown("""
    <div class="glass-card" style="margin-bottom: 2rem;">
        <h3 style="color: rgba(255, 255, 255, 0.9); margin-bottom: 1rem; text-align: center;">⚙️ Technical Specifications</h3>
        <div style="color: rgba(255, 255, 255, 0.8); line-height: 1.8;">
            <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                <strong>📡 Data Source:</strong> Yahoo Finance API
            </div>
            <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                <strong>🔧 Feature Engineering:</strong> 34 technical indicators and cross-asset features
            </div>
            <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                <strong>📏 Sequence Length:</strong> 128 hours (5+ days) of historical data
            </div>
            <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                <strong>📚 Training Period:</strong> 2+ years of hourly data
            </div>
            <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                <strong>⚡ Update Frequency:</strong> Real-time predictions
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: rgba(255, 255, 255, 0.9); margin-bottom: 1rem; text-align: center;">📈 Backtest Performance</h3>
        <div style="color: rgba(255, 255, 255, 0.8); line-height: 1.8;">
            <div style="background: linear-gradient(45deg, rgba(0, 255, 136, 0.2), rgba(46, 204, 113, 0.2)); padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border: 1px solid rgba(0, 255, 136, 0.3);">
                <strong>💰 Total Return:</strong> <span style="color: #00ff88; font-weight: 900;">239.97%</span>
            </div>
            <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                <strong>🏦 Start Capital:</strong> $10,000.00
            </div>
            <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                <strong>🎯 Final Capital:</strong> $33,997.15
            </div>
            <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                <strong>📊 Total Trades:</strong> 8,944
            </div>
            <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                <strong>⚡ Avg Hourly PnL:</strong> $2.68 ± $91.17
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(255, 107, 107, 0.1); border: 2px solid rgba(255, 107, 107, 0.3); border-radius: 15px; padding: 1.5rem; margin-top: 2rem; backdrop-filter: blur(10px);">
        <div style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 1rem;">⚠️</div>
            <h4 style="color: #ff6b6b; margin-bottom: 1rem;">Important Disclaimer</h4>
            <p style="color: rgba(255, 255, 255, 0.9); line-height: 1.6; margin: 0;">
                This tool is for <strong>educational and research purposes only</strong>.<br>
                Past performance is <strong>not indicative of future results</strong>.<br>
                Always consult with financial advisors before making investment decisions.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
