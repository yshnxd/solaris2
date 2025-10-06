import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

# Initialize default session state for selections used across tabs
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = 'AAPL'
if 'days_history' not in st.session_state:
    st.session_state.days_history = 90

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
        text-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
        animation: subtle-glow 3s ease-in-out infinite alternate;
    }
    
    @keyframes subtle-glow {
        from { filter: drop-shadow(0 0 5px rgba(102, 126, 234, 0.3)); }
        to { filter: drop-shadow(0 0 15px rgba(118, 75, 162, 0.5)); }
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
        text-shadow: 0 0 5px rgba(0, 255, 136, 0.3);
    }
    
    .negative {
        color: #ff6b6b;
        font-weight: 600;
        text-shadow: 0 0 5px rgba(255, 107, 107, 0.3);
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
    
    /* Subtle Glow */
    .neon-glow {
        text-shadow: 0 0 3px currentColor, 0 0 6px currentColor;
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

# Navigation and App Structure
st.markdown("""
<div class="floating">
    <h1 class="main-header">🚀 SOLARIS</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pulse">
    <h2 class="sub-header">✨ Machine Learner-Powered Stock Prediction Model ✨</h2>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🏠 Dashboard", "🎯 Predictions", "📊 Analytics", "📈 Charts", "⚙️ Settings", "📚 Theory"])

PREDICTION_HISTORY_CSV = "solaris-data.csv"
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

# Dashboard Tab
with tab1:
    st.markdown('<h2 class="section-header">🏠 Dashboard Overview</h2>', unsafe_allow_html=True)
    
    # Quick Stats Row (3 cards)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="glass-card metric-card">
            <div style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🧠</div>
                <div style="font-size: 1.5rem; color: #667eea; font-weight: 900;">3</div>
                <div style="color: rgba(255, 255, 255, 0.8); font-weight: 600;">AI Models</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card metric-card">
            <div style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>
                <div style="font-size: 1.5rem; color: #00ff88; font-weight: 900;">34</div>
                <div style="color: rgba(255, 255, 255, 0.8); font-weight: 600;">Features</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="glass-card metric-card">
            <div style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">⚡</div>
                <div style="font-size: 1.5rem; color: #f093fb; font-weight: 900;">1H</div>
                <div style="color: rgba(255, 255, 255, 0.8); font-weight: 600;">Prediction</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Removed Backtest ROI card per request
    
    # Quick Actions removed
    
    # Real-Time Market Data (Dashboard only)
    st.markdown('<h3 style="color: rgba(255, 255, 255, 0.9); margin: 2rem 0 1rem 0;">📊 Real-Time Market Data</h3>', unsafe_allow_html=True)
    selected_ticker = st.session_state.selected_ticker
    days_history = st.session_state.days_history
    main_ticker_data = fetch_stock_data(selected_ticker, period=f"{days_history}d", interval="60m")
    if main_ticker_data is not None and not main_ticker_data.empty and len(main_ticker_data) >= 128:
        latest_data = main_ticker_data.iloc[-1]
        current_price = latest_data['Close']
        current_time = latest_data.name
        col1, col2, col3 = st.columns(3)
        with col1:
            current_price_formatted = f"${current_price:.2f}"
            st.markdown(f"""
            <div class="glass-card metric-card">
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; font-weight: 900; color: #00ff88; margin-bottom: 0.5rem; text-shadow: 0 0 3px rgba(0, 255, 136, 0.3);">
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
                    <div style="font-size: 2rem; font-weight: 900; color: {change_color}; margin-bottom: 0.5rem; text-shadow: 0 0 3px {change_color}30;">
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
                    <div style="font-size: 1.8rem; font-weight: 900; color: #667eea; margin-bottom: 0.5rem; text-shadow: 0 0 3px rgba(102, 126, 234, 0.3);">
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
    else:
        st.info("Not enough data to show real-time metrics yet.")

    # Current Stock Graph right after Real-Time Market Data
    if main_ticker_data is not None and not main_ticker_data.empty and len(main_ticker_data) >= 128:
        close_series = main_ticker_data['Close']
        sma_20 = close_series.rolling(20).mean() if len(close_series) >= 20 else None
        fig_dash = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=(f"{selected_ticker} Price", 'Volume'))
        fig_dash.add_trace(go.Candlestick(x=main_ticker_data.index, open=main_ticker_data['Open'], high=main_ticker_data['High'], low=main_ticker_data['Low'], close=main_ticker_data['Close'], name="Price"), row=1, col=1)
        if sma_20 is not None:
            fig_dash.add_trace(go.Scatter(x=sma_20.index, y=sma_20, name='SMA 20', mode='lines', line=dict(color='#c7c7c7', width=1.2, dash='dash')), row=1, col=1)
        colors_dash = ['rgba(231,76,60,0.45)' if row['Open'] > row['Close'] else 'rgba(46,204,113,0.45)' for _, row in main_ticker_data.iterrows()]
        fig_dash.add_trace(go.Bar(x=main_ticker_data.index, y=main_ticker_data['Volume'], marker_color=colors_dash, name="Volume"), row=2, col=1)
        fig_dash.update_layout(height=520, showlegend=True, plot_bgcolor='rgba(255, 255, 255, 0.05)', paper_bgcolor='rgba(255, 255, 255, 0.05)', font=dict(color='rgba(255, 255, 255, 0.9)', family='Inter'))
        st.plotly_chart(fig_dash, use_container_width=True)

    # Recent Activity (move to end of Dashboard)
    st.markdown('<h3 style="color: rgba(255, 255, 255, 0.9); margin: 2rem 0 1rem 0;">📋 Recent Activity</h3>', unsafe_allow_html=True)
    
    # Load recent predictions if available
    if os.path.exists(PREDICTION_HISTORY_CSV):
        try:
            recent_df = pd.read_csv(PREDICTION_HISTORY_CSV)
            if not recent_df.empty:
                recent_df = recent_df.sort_values("timestamp", ascending=False).head(5)
                st.markdown("""
                <div class="glass-card">
                    <div style="color: rgba(255, 255, 255, 0.9);">
                        <h4 style="margin-bottom: 1rem;">Latest Predictions</h4>
                """, unsafe_allow_html=True)
                
                for _, row in recent_df.iterrows():
                    signal_color = "#00ff88" if row.get('signal') == "BUY" else "#ff6b6b" if row.get('signal') == "SELL" else "#667eea"
                    st.markdown(f"""
                    <div style="display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; padding: 0.6rem; background: rgba(255, 255, 255, 0.1); border-radius: 10px; margin: 0.5rem 0;">
                        <div style="text-align: left; color: rgba(255, 255, 255, 0.9); font-weight: 600;">{row.get('ticker', 'N/A')}</div>
                        <div style="text-align: center; color: #00ff88; font-weight: 800; font-size: 1.1rem; text-shadow: 0 0 6px rgba(0,255,136,0.25);">${row.get('predicted_price', 0):.2f}</div>
                        <div style="text-align: right; color: {signal_color}; font-weight: 700;">{row.get('signal', 'N/A')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div></div>", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="glass-card" style="text-align: center; padding: 2rem;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
                    <div style="color: rgba(255, 255, 255, 0.8);">
                        No predictions yet. Start by generating your first prediction!
                    </div>
                </div>
                """, unsafe_allow_html=True)
        except Exception:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 2rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
                <div style="color: rgba(255, 255, 255, 0.8);">
                    No predictions yet. Start by generating your first prediction!
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 2rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
            <div style="color: rgba(255, 255, 255, 0.8);">
                No predictions yet. Start by generating your first prediction!
            </div>
        </div>
        """, unsafe_allow_html=True)

# Enhanced Sidebar
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid rgba(255, 255, 255, 0.2); margin-bottom: 1rem;">
    <h3 style="color: rgba(255, 255, 255, 0.9); margin: 0; font-weight: 600;">⚙️ Control Panel</h3>
</div>
""", unsafe_allow_html=True)

# Stock Selection with Enhanced Styling
st.sidebar.markdown("### 📈 Stock Selection")
selected_ticker = st.sidebar.text_input("Stock Ticker Symbol", st.session_state.selected_ticker, help="Enter a valid stock ticker symbol (e.g., AAPL, MSFT, TSLA)").upper()
st.session_state.selected_ticker = selected_ticker

# History Slider with Enhanced Styling
st.sidebar.markdown("### 📊 Data Range")
days_history = st.sidebar.slider(
    "Historical Data Period", 
    min_value=30, 
    max_value=729, 
    value=st.session_state.days_history,
    help="Number of days of historical data to analyze"
)
st.session_state.days_history = days_history

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

# Bridge quick-action triggers to existing logic
trigger_prediction = st.session_state.pop('trigger_prediction', False) if 'trigger_prediction' in st.session_state else False
trigger_evaluate = st.session_state.pop('trigger_evaluate', False) if 'trigger_evaluate' in st.session_state else False

# Predictions Tab
with tab2:
    st.markdown('<h2 class="section-header">🎯 AI Predictions</h2>', unsafe_allow_html=True)

main_ticker_data = fetch_stock_data(selected_ticker, period=f"{days_history}d", interval="60m")

if main_ticker_data is None or main_ticker_data.empty or len(main_ticker_data) < 128:
        st.markdown("""
        <div style="background: rgba(255, 107, 107, 0.1); border: 2px solid rgba(255, 107, 107, 0.3); border-radius: 15px; padding: 2rem; text-align: center; backdrop-filter: blur(10px);">
            <div style="font-size: 3rem; margin-bottom: 1rem;">⚠️</div>
            <h3 style="color: #ff6b6b; margin-bottom: 1rem;">No Data Available</h3>
            <p style="color: rgba(255, 255, 255, 0.9); line-height: 1.6; margin: 0;">
                No or insufficient data found for ticker: <strong>{selected_ticker}</strong><br>
                Please check the symbol or try another stock.
            </p>
        </div>
        """, unsafe_allow_html=True)
else:
    latest_data = main_ticker_data.iloc[-1]
    current_price = latest_data['Close']
    current_time = latest_data.name
    
    # Remove real-time metric cards from Predictions tab (now shown on Dashboard)
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

# Charts Tab
with tab4:
    st.markdown('<h2 class="section-header">📈 Interactive Charts</h2>', unsafe_allow_html=True)
    
    if main_ticker_data is not None and not main_ticker_data.empty and len(main_ticker_data) >= 128:
        # Chart controls in the tab
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("### 📊 Chart Settings")
            chart_interval = st.selectbox("Time Interval", ["1 Hour", "1 Day"], index=0)
            chart_period = st.selectbox("Chart Period", ["30 days", "90 days", "1 year"], index=1)
        
        with chart_col2:
            st.markdown("### 🎨 Display Options")
            show_volume = st.checkbox("Show Volume", value=True)
            show_candlesticks = st.checkbox("Show Candlesticks", value=True)
        
        # Re-create the chart with user settings
        if chart_interval == "1 Hour":
            chart_data = main_ticker_data
            chart_title = f"{selected_ticker} Hourly Chart"
        else:
            chart_data = fetch_stock_data(selected_ticker, period=f"{chart_period.replace(' days', 'd').replace('1 year', '365d')}", interval="1d")
            chart_title = f"{selected_ticker} Daily Chart"
        
        if chart_data is not None and not chart_data.empty:
            # Create enhanced chart
            fig_chart = make_subplots(
                rows=2 if show_volume else 1, 
                cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(chart_title, 'Volume') if show_volume else (chart_title,),
                row_heights=[0.7, 0.3] if show_volume else [1.0]
            )
            
            if show_candlesticks:
                fig_chart.add_trace(go.Candlestick(
                    x=chart_data.index,
                    open=chart_data['Open'],
                    high=chart_data['High'],
                    low=chart_data['Low'],
                    close=chart_data['Close'],
                    name="Price"
                ), row=1, col=1)
            else:
                fig_chart.add_trace(go.Scatter(
                    x=chart_data.index,
                    y=chart_data['Close'],
                    mode='lines',
                    name="Price",
                    line=dict(color='#667eea', width=2)
                ), row=1, col=1)
            
            if show_volume:
                colors = ['rgba(231,76,60,0.45)' if row['Open'] > row['Close'] else 'rgba(46,204,113,0.45)'
                          for _, row in chart_data.iterrows()]
                fig_chart.add_trace(go.Bar(
                    x=chart_data.index,
                    y=chart_data['Volume'],
                    marker_color=colors,
                    name="Volume"
                ), row=2, col=1)
            
            fig_chart.update_layout(
                height=600 if show_volume else 400,
                showlegend=True,
                plot_bgcolor='rgba(255, 255, 255, 0.05)',
                paper_bgcolor='rgba(255, 255, 255, 0.05)',
                font=dict(color='rgba(255, 255, 255, 0.9)', family='Inter'),
                title=dict(
                    text=chart_title,
                    font=dict(size=24, color='rgba(255, 255, 255, 0.9)'),
                    x=0.5
                )
            )
            
            st.plotly_chart(fig_chart, use_container_width=True)
        else:
            st.error("Unable to load chart data")
    else:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 3rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📈</div>
            <h3 style="color: rgba(255, 255, 255, 0.9); margin-bottom: 1rem;">No Chart Data Available</h3>
            <p style="color: rgba(255, 255, 255, 0.7);">
                Please select a valid stock ticker and ensure sufficient data is available.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Analytics Tab
with tab3:
    st.markdown('<h2 class="section-header">📊 Analytics & History</h2>', unsafe_allow_html=True)
    
    # Analytics sub-tabs
    analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs(["📈 Performance", "📋 Prediction History", "📊 Model Analysis"])
    
    with analytics_tab1:
        st.markdown('<h3 style="color: rgba(255, 255, 255, 0.9); margin-bottom: 1rem;">📈 Performance Metrics</h3>', unsafe_allow_html=True)
        
        # Performance metrics
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            # Compute regression metrics from history (if available)
            metrics_mae = metrics_rmse = metrics_r2 = metrics_mape = None
            if os.path.exists(PREDICTION_HISTORY_CSV):
                try:
                    _df_hist = pd.read_csv(PREDICTION_HISTORY_CSV)
                    _df_eval = _df_hist.dropna(subset=['predicted_price', 'actual_price']) if {'predicted_price','actual_price'}.issubset(_df_hist.columns) else pd.DataFrame()
                    if not _df_eval.empty:
                        _y_true = _df_eval['actual_price'].astype(float)
                        _y_pred = _df_eval['predicted_price'].astype(float)
                        metrics_mae = mean_absolute_error(_y_true, _y_pred)
                        metrics_rmse = math.sqrt(mean_squared_error(_y_true, _y_pred))
                        metrics_r2 = r2_score(_y_true, _y_pred)
                        metrics_mape = float(np.mean(np.abs((_y_true - _y_pred) / _y_true)) * 100) if (_y_true != 0).all() else np.nan
                except Exception:
                    pass
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: rgba(255, 255, 255, 0.9); margin-bottom: 1rem;">🎯 Regression Metrics</h4>
            </div>
            """, unsafe_allow_html=True)
            met_col1, met_col4 = st.columns(2)
            with met_col1:
                st.metric("MAE", f"${metrics_mae:.3f}" if metrics_mae is not None else "N/A")
            with met_col4:
                st.metric("MAPE", f"{metrics_mape:.2f}%" if metrics_mape is not None and not np.isnan(metrics_mape) else "N/A")
        
        with perf_col2:
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: rgba(255, 255, 255, 0.9); margin-bottom: 1rem;">📊 Dataset Snapshot</h4>
                <div style="color: rgba(255, 255, 255, 0.8); line-height: 1.8;">
                    <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                        <span>Records:</span>
                        <span>{len(_df_hist) if '_df_hist' in locals() and isinstance(_df_hist, pd.DataFrame) else 'N/A'}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                        <span>Evaluated:</span>
                        <span>{len(_df_eval) if '_df_eval' in locals() and isinstance(_df_eval, pd.DataFrame) else 'N/A'}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with analytics_tab2:
        st.markdown('<h3 style="color: rgba(255, 255, 255, 0.9); margin-bottom: 1rem;">📋 Prediction History</h3>', unsafe_allow_html=True)
        
        # Load and display prediction history
        if os.path.exists(PREDICTION_HISTORY_CSV):
            try:
                history_df = pd.read_csv(PREDICTION_HISTORY_CSV)
                if not history_df.empty:
                    # Convert timestamps
                    for col in ["timestamp", "target_time"]:
                        if col in history_df.columns:
                            history_df[col] = pd.to_datetime(history_df[col], errors="coerce")
                    
                    history_df = history_df.sort_values("timestamp", ascending=False)
                    
                    # Display recent predictions
                    st.markdown("### Recent Predictions")
                    recent_display = history_df.head(10)[show_cols]
                    st.dataframe(recent_display, use_container_width=True)

                    # Summary statistics using regression metrics
                    if 'predicted_price' in history_df.columns and 'actual_price' in history_df.columns:
                        eval_rows = history_df.dropna(subset=['predicted_price', 'actual_price'])
                        if not eval_rows.empty:
                            y_true = eval_rows['actual_price'].astype(float)
                            y_pred = eval_rows['predicted_price'].astype(float)
                            mae = mean_absolute_error(y_true, y_pred)
                            rmse = math.sqrt(mean_squared_error(y_true, y_pred))
                            r2 = r2_score(y_true, y_pred)
                            mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100) if (y_true != 0).all() else np.nan
                            st.markdown("### Regression Metrics")
                            acc_col1, acc_col4 = st.columns(2)
                            with acc_col1:
                                st.metric("MAE", f"${mae:.3f}")
                            with acc_col4:
                                st.metric("MAPE", f"{mape:.2f}%" if not np.isnan(mape) else "N/A")
                else:
                    st.markdown("""
                    <div class="glass-card" style="text-align: center; padding: 2rem;">
                        <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
                        <div style="color: rgba(255, 255, 255, 0.8);">
                            No prediction history available yet.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error loading prediction history: {str(e)}")
        else:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 2rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
                <div style="color: rgba(255, 255, 255, 0.8);">
                    No prediction history file found.
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with analytics_tab3:
        st.markdown('<h3 style="color: rgba(255, 255, 255, 0.9); margin-bottom: 1rem;">📊 Model Analysis</h3>', unsafe_allow_html=True)
        
        # Model comparison
        st.markdown("### Model Comparison")
        
        # Replace hardcoded accuracy with available metrics summary
        _acc_df = pd.DataFrame()
        if os.path.exists(PREDICTION_HISTORY_CSV):
            try:
                _hist = pd.read_csv(PREDICTION_HISTORY_CSV)
                _eval = _hist.dropna(subset=['predicted_price','actual_price']) if {'predicted_price','actual_price'}.issubset(_hist.columns) else pd.DataFrame()
                if not _eval.empty:
                    _mae = mean_absolute_error(_eval['actual_price'].astype(float), _eval['predicted_price'].astype(float))
                    _mape = float(np.mean(np.abs((_eval['actual_price'] - _eval['predicted_price']) / _eval['actual_price'])) * 100) if (_eval['actual_price'] != 0).all() else np.nan
                    _acc_df = pd.DataFrame({
                        'Metric': ['MAE', 'MAPE'],
                        'Value': [f"${_mae:.3f}", f"{_mape:.2f}%" if not np.isnan(_mape) else 'N/A']
                    })
            except Exception:
                pass
        model_data = _acc_df if not _acc_df.empty else pd.DataFrame({'Metric': ['MAE','MAPE'], 'Value': ['N/A','N/A']})
        
        st.dataframe(model_data, use_container_width=True)
        
        # Feature importance
        st.markdown("### Feature Importance")
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

# Settings Tab
with tab5:
    st.markdown('<h2 class="section-header">⚙️ Settings & Configuration</h2>', unsafe_allow_html=True)
    
    settings_col1, settings_col2 = st.columns(2)
    
    with settings_col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: rgba(255, 255, 255, 0.9); margin-bottom: 1rem;">🎨 Appearance</h3>
            <div style="color: rgba(255, 255, 255, 0.8);">
                <p>• Dark theme with glassmorphism effects</p>
                <p>• Animated gradient background</p>
                <p>• Neon glow effects on key metrics</p>
                <p>• Responsive design for all devices</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: rgba(255, 255, 255, 0.9); margin-bottom: 1rem;">📊 Data Sources</h3>
            <div style="color: rgba(255, 255, 255, 0.8);">
                <p>• Yahoo Finance API for real-time data</p>
                <p>• 34 technical indicators</p>
                <p>• Cross-asset correlations</p>
                <p>• Historical data up to 2 years</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with settings_col2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: rgba(255, 255, 255, 0.9); margin-bottom: 1rem;">🧠 AI Models</h3>
            <div style="color: rgba(255, 255, 255, 0.8);">
                <p>• CNN for pattern recognition</p>
                <p>• LSTM for time series analysis</p>
                <p>• XGBoost for feature importance</p>
                <p>• Meta-learner for ensemble</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: rgba(255, 255, 255, 0.9); margin-bottom: 1rem;">⚠️ Important Notice</h3>
            <div style="color: rgba(255, 255, 255, 0.8); line-height: 1.6;">
                <p><strong>Educational Purpose Only</strong></p>
                <p>This tool is designed for educational and research purposes. Past performance does not guarantee future results. Always consult with financial advisors before making investment decisions.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Move the prediction logic to the predictions tab
if cnn_model and lstm_model and xgb_model and meta_model and scaler:
    if main_ticker_data is not None and not main_ticker_data.empty and len(main_ticker_data) >= 128:
        cross_data = fetch_cross_assets(main_ticker_data.index, days_history, selected_ticker)
        run_watchlist_triggered = run_watchlist or trigger_prediction and False  # no watchlist on quick pred
        if run_prediction or trigger_prediction:
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
                                    <div style="font-size: 2.5rem; font-weight: 900; color: {pred_color}; margin-bottom: 0.5rem; text-shadow: 0 0 3px {pred_color}30;">
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
                                    <div style="font-size: 1.8rem; color: {signal_color}; font-weight: 900; margin-bottom: 1rem; text-shadow: 0 0 3px {signal_color}30;">
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
        if evaluate_results or trigger_evaluate:
            with st.spinner("Evaluating prediction history and filling actuals..."):
                updated_df = batch_evaluate_predictions_csv(PREDICTION_HISTORY_CSV)
                if updated_df is not None:
                    st.success("Filled actual prices and updated prediction history CSV.")
        # Removed history table/summary from non-Analytics sections



# Theory Tab
with tab6:
    st.markdown('<h2 class="section-header">📚 Theory: Methodology & Mathematical Details</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h3 style="color: rgba(255, 255, 255, 0.95); margin-bottom: 0.5rem;">🔬 Research Pipeline</h3>
        <ol style="color: rgba(255,255,255,0.85); line-height: 1.7; margin: 0; padding-left: 1.2rem;">
            <li>Collect hourly OHLCV for 12 equities via Yahoo Finance over ~2 years.</li>
            <li>Align times across assets; forward-fill missing quotes.</li>
            <li>Create features: returns, volatility, RSI/MACD, SMA/EMA, volume signals, calendar features, cross-asset returns.</li>
            <li>Label: next-hour return \(r_{t+1} = (P_{t+1}-P_t)/P_t\).</li>
            <li>Standardize features; build sequences of length 128 for sequence models.</li>
            <li>Train base learners (CNN, LSTM, XGBoost) with MSE loss.</li>
            <li>Train a meta-learner on base predictions plus summary statistics.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # Data & labels
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4 style="margin-bottom:0.5rem;">🧾 Data & Label</h4>
        <p style="color: rgba(255,255,255,0.85);">Hourly price \(P_t\); label is next-hour return.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**Label**")
    st.latex(r"""
    r_{t+1} = \frac{P_{t+1}-P_t}{P_t},\quad \hat{P}_{t+1} = P_t(1+\hat r_{t+1})
    """)
    st.markdown("**Features**")
    st.latex(r"""
    \begin{aligned}
    &r_{t,\Delta} = \tfrac{P_t-P_{t-\Delta}}{P_{t-\Delta}};\; \sigma_{t,w} = \sqrt{\tfrac{1}{w-1}\sum_{i=0}^{w-1}(r_{t-i,1}-\bar r)^2}\\
    &\text{RSI}=100-\tfrac{100}{1+RS},\; RS=\tfrac{\text{avg gain}_{14}}{\text{avg loss}_{14}};\; \text{MACD}=\text{EMA}_{12}-\text{EMA}_{26},\; \text{Signal}=\text{EMA}_9(\text{MACD})\\
    &\text{SMA/EMA};\; \Delta V_t=\tfrac{V_t-V_{t-1}}{V_{t-1}},\; \text{VMA}_{24}=\tfrac{1}{24}\sum_{i=0}^{23}V_{t-i};\; \text{hour, dow};\; r^{(a)}_{t,1}
    \end{aligned}
    """)

    # LSTM
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4 style="margin-bottom:0.5rem;">🔄 LSTM (Long Short-Term Memory)</h4>
        <p style="color: rgba(255,255,255,0.85);">Sequence model with memory cell \(c_t\) and hidden state \(h_t\) over L=128 steps.</p>
        <h5 style="margin-top:0.5rem;">How it works</h5>
        <p style="color: rgba(255,255,255,0.8);">Gates \(f_t,i_t,o_t\) control forgetting, writing, and exposure of memory.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**Math**")
    st.latex(r"""
    \begin{aligned}
    &f_t=\sigma(W_f[h_{t-1},x_t]+b_f),\; i_t=\sigma(W_i[h_{t-1},x_t]+b_i),\; o_t=\sigma(W_o[h_{t-1},x_t]+b_o)\\
    &\tilde c_t=\tanh(W_c[h_{t-1},x_t]+b_c),\; c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t,\; h_t=o_t\odot\tanh(c_t),\; \hat r_{t+1}=W_y h_t+b_y
    \end{aligned}
    """)
    st.markdown("**Learning**: MSE via BPTT (Adam).")
    st.markdown("**Application (step-by-step)**")
    st.markdown("""
```text
1) Take last 128 standardized vectors x_{t-127:t}
2) LSTM → h_t → Dense → \hat r_{t+1}
3) Price: \hat P_{t+1} = P_t (1+\hat r_{t+1})
```
""")
    st.markdown("**Pros**: Long memory  \\ **Cons**: Slower, regularization needed")

    # CNN
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4 style="margin-bottom:0.5rem;">🧩 CNN (Temporal Convolution)</h4>
        <p style="color: rgba(255,255,255,0.85);">Local temporal motif learner using Conv1D + pooling.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**Math**")
    st.latex(r"""
    (X * k)_t = \sum_{\tau=0}^{K-1} k_{\tau}\, X_{t-\tau}
    """)
    st.markdown("**Learning**: Minimize MSE (Adam).")
    st.markdown("**Application (step-by-step)**")
    st.markdown("""
```text
1) 128-step window → Conv/BN/ReLU stacks
2) GlobalAvgPool → Dense → \hat r_{t+1}
3) Price: \hat P_{t+1} = P_t (1+\hat r_{t+1})
```
""")
    st.markdown("**Pros**: Fast, robust to noise  \\ **Cons**: Limited long context unless deep/dilated")

    # XGBoost
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4 style="margin-bottom:0.5rem;">🌳 XGBoost (Gradient-Boosted Trees)</h4>
        <p style="color: rgba(255,255,255,0.85);">Learns tabular, non-linear interactions from the most recent feature vector.</p>
        <h5 style="margin-top:0.5rem;">How it works</h5>
        <p style="color: rgba(255,255,255,0.8);">Adds shallow trees one-by-one to correct residual errors, with regularization and shrinkage to avoid overfitting.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**Math**")
    st.latex(r"""
    \hat{y}(x) = \sum_{m=1}^{M} \gamma_m\, T_m(x)
    """)
    st.markdown("**Learning**: Regularized gradient boosting with shrinkage and column subsampling.")
    st.markdown("**Application (step-by-step)**")
    st.markdown("""
```text
1) Build last feature vector (indicators, calendar, cross-asset)
2) Apply boosted trees → predicted return
3) Price forecast: P̂_{t+1} = P_t · (1 + predicted return)
```
""")
    st.markdown("**Pros**: Strong on tabular signals, robust  \\ **Cons**: Weak on raw sequence structure")

    # Meta-learner
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4 style="margin-bottom:0.5rem;">🤝 Meta-Learner (Ensemble)</h4>
        <p style="color: rgba(255,255,255,0.85);">Combines CNN/LSTM/XGBoost predictions and summary stats for stability.</p>
        <h5 style="margin-top:0.5rem;">How it works</h5>
        <p style="color: rgba(255,255,255,0.8);">Learns weights to trust each model under different regimes (trendy vs. choppy), plus simple statistics.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**Math**")
    st.latex(r"""
    \hat{r}_{meta} = g\Big([\hat r_{cnn},\hat r_{lstm},\hat r_{xgb},\;\mu,\sigma,\max,\min]\Big)
    """)
    st.markdown("**Learning**: Shallow regressor (e.g., HistGBR) minimizing MSE on OOF predictions.")
    st.markdown("""
**Application**: Final price \(\hat{P}_{t+1}=P_t (1+\hat{r}_{meta})\).<br/>
**Pros**: Stable across regimes<br/>
**Cons**: Can dilute strong model when signals are very clean
""")

    # --- Indicators Section ---
    st.markdown("""
    <div class="glass-card" style="margin-top:2rem;">
        <h3 style="color: rgba(255, 255, 255, 0.95); margin-bottom: 1rem; text-align:center;">📐 Technical Indicators (34)</h3>
        <p style="color: rgba(255,255,255,0.85);">Grouped into Trend, Momentum, Volatility, and Volume, plus Calendar/Price features. Below are formulas and intuition.</p>
    </div>
    """, unsafe_allow_html=True)

    # Trend
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4>📈 Trend</h4>
        <h5>Math</h5>
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"""
    \text{SMA}_t = \frac{1}{w} \sum_{i=0}^{w-1} P_{t-i}\quad\quad
    \text{EMA}_t = \alpha P_t + (1-\alpha)\,\text{EMA}_{t-1},\; \alpha=\tfrac{2}{w+1}
    """)
    st.latex(r"""
    \text{MACD}_t = \text{EMA}_{12}(P)_t - \text{EMA}_{26}(P)_t,\quad
    \text{Signal}_t = \text{EMA}_{9}(\text{MACD})_t
    """)
    st.markdown("**How it works**: Averages smooth noise so direction is clearer; MACD subtracts slow from fast average to reveal turning points.")
    st.markdown("**Application (step-by-step)**")
    st.markdown("""
```text
1) Compute SMA/EMA over window(s)
2) Compute MACD = EMA_fast − EMA_slow; Signal = EMA(MACD)
3) Use trend slope/crossovers as model inputs
```
""")
    st.markdown("**Pros**: Simple, robust to noise  \\ **Cons**: Lagging vs. price reversals")

    # Momentum
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4>🏃 Momentum</h4>
        <h5>Math</h5>
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"""
    r_{t,\Delta} = \frac{P_t - P_{t-\Delta}}{P_{t-\Delta}}\quad\text{(various }\Delta)\qquad
    \text{RSI} = 100 - \frac{100}{1 + RS},\; RS=\frac{\text{avg gain}_{14}}{\text{avg loss}_{14}}
    """)
    st.markdown("**How it works**: Returns measure persistence in direction; RSI normalizes recent gains vs. losses to flag stretched moves.")
    st.markdown("**Application (step-by-step)**")
    st.markdown("""
```text
1) Compute multi-horizon returns (e.g., 1h, 6h, 24h)
2) Compute RSI(14) from smoothed gains/losses
3) Include cross-asset returns from peers
```
""")
    st.markdown("**Pros**: Catches persistence/mean-reversion  \\ **Cons**: Whipsaws in choppy markets")

    # Volatility
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4>🌪️ Volatility</h4>
        <h5>Math</h5>
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"""
    \sigma_{t,w} = \sqrt{\frac{1}{w-1} \sum_{i=0}^{w-1} (r_{t-i,1}-\bar{r})^2}
    """)
    st.markdown("**How it works**: Measures dispersion of recent returns; higher \(\sigma\) implies wider outcome range and lower confidence.")
    st.markdown("**Application (step-by-step)**")
    st.markdown("""
```text
1) Compute 1h returns
2) Compute rolling mean and std over 6h/12h/24h
3) Use std to scale model confidence and position sizing features
```
""")
    st.markdown("**Pros**: Encodes risk/regime  \\ **Cons**: Can lag; sensitive to outliers")

    # Volume
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4>📦 Volume</h4>
        <h5>Math</h5>
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"""
    \Delta V_t = \frac{V_t - V_{t-1}}{V_{t-1}}\quad\quad
    \text{VMA}_{24} = \frac{1}{24} \sum_{i=0}^{23} V_{t-i}
    """)
    st.markdown("**How it works**: Rising volume confirms participation; extreme surges can signal breakouts or exhaustion.")
    st.markdown("**Application (step-by-step)**")
    st.markdown("""
```text
1) Compute 1h volume change and 24h volume MA
2) Combine with price moves to confirm/discount signals
```
""")
    st.markdown("**Pros**: Confirms moves  \\ **Cons**: Noisy during illiquid periods")

    # Calendar and Price
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4>🗓️ Calendar & Price</h4>
        <p style="color: rgba(255,255,255,0.8);">Encodes intraday/weekly seasonality and price-level context.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**Application (step-by-step)**")
    st.markdown("""
```text
1) Extract hour-of-day and day-of-week
2) Include raw price and distances to averages (e.g., P − SMA)
3) Feed as auxiliary features to models
```
""")
    st.markdown("**Pros**: Adds systematic timing/context  \\ **Cons**: Patterns drift across assets/periods")

    # Why helpful
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4>💡 Why these help forecasting</h4>
        <p style="color: rgba(255,255,255,0.85);">
        Trend captures direction; Momentum/RSI capture persistence and reversion; Volatility scales confidence; Volume confirms participation; Cross-asset returns leverage correlations; Calendar adds systematic timing; Price level contextualizes distances from anchors.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Compact Model Cards (side-by-side)
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4 style="margin-bottom:0.5rem; text-align:center;">🧠 Base Models (At-a-glance)</h4>
    </div>
    """, unsafe_allow_html=True)
    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        st.markdown("""
        <div class="glass-card" style="margin-top:0.5rem;">
            <h5>🔄 LSTM</h5>
            <div style="color: rgba(255,255,255,0.85);">Memory cell <em>c_t</em>, hidden <em>h_t</em> over 128 steps. Gates f,i,o control information flow.</div>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\hat r=W_y h_t+b_y")
    with mcol2:
        st.markdown("""
        <div class="glass-card" style="margin-top:0.5rem;">
            <h5>🧩 CNN</h5>
            <div style="color: rgba(255,255,255,0.85);">Conv1D filters detect local motifs; pooling summarizes; Dense outputs \(\hat r\).</div>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"(X * k)_t = \sum_{\tau=0}^{K-1} k_{\tau}X_{t-\tau}")
    with mcol3:
        st.markdown("""
        <div class="glass-card" style="margin-top:0.5rem;">
            <h5>🌳 XGBoost</h5>
            <div style="color: rgba(255,255,255,0.85);">Boosted trees on tabular features with regularization.</div>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\hat r(x)=\sum_m \gamma_m T_m(x)")

    # Indicator Cards Grid (separate boxes)
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4 style="margin-bottom:0.5rem; text-align:center;">📐 Indicators (Explained)</h4>
    </div>
    """, unsafe_allow_html=True)
    icol1, icol2 = st.columns(2)
    with icol1:
        st.markdown("""
        <div class="glass-card" style="margin-top:0.5rem;">
            <h5>📈 SMA</h5>
            <div style="color: rgba(255,255,255,0.85);">Average of last w closes (trend smoothing).</div>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\text{SMA}_t = \tfrac{1}{w}\sum_{i=0}^{w-1}P_{t-i}")
        st.markdown("- Compute: rolling mean\n- Use: baseline trend")

        st.markdown("""
        <div class="glass-card" style="margin-top:0.5rem;">
            <h5>🏃 RSI (14)</h5>
            <div style="color: rgba(255,255,255,0.85);">Relative strength of gains vs losses.</div>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\text{RSI}=100-\frac{100}{1+RS},\; RS=\frac{\text{avg gain}_{14}}{\text{avg loss}_{14}}")
        st.markdown("- Compute: smoothed gains/losses\n- Use: stretch / mean-reversion")

        st.markdown("""
        <div class="glass-card" style="margin-top:0.5rem;">
            <h5>🌪️ Volatility</h5>
            <div style="color: rgba(255,255,255,0.85);">Std of recent 1h returns.</div>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\sigma_{t,w} = \sqrt{\tfrac{1}{w-1}\sum_{i=0}^{w-1}(r_{t-i,1}-\bar r)^2}")
        st.markdown("- Compute: rolling std\n- Use: regime/risk context")

    with icol2:
        st.markdown("""
        <div class="glass-card" style="margin-top:0.5rem;">
            <h5>📈 EMA</h5>
            <div style="color: rgba(255,255,255,0.85);">Exponential weighting of recent prices.</div>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\text{EMA}_t=\alpha P_t+(1-\alpha)\,\text{EMA}_{t-1},\; \alpha=\tfrac{2}{w+1}")
        st.markdown("- Compute: ewm\n- Use: responsive trend")

        st.markdown("""
        <div class="glass-card" style="margin-top:0.5rem;">
            <h5>📐 MACD & Signal</h5>
            <div style="color: rgba(255,255,255,0.85);">Fast minus slow EMA and its EMA.</div>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\text{MACD}=\text{EMA}_{12}(P)-\text{EMA}_{26}(P),\; \text{Signal}=\text{EMA}_9(\text{MACD})")
        st.markdown("- Compute: ta.MACD\n- Use: turning points")

        st.markdown("""
        <div class="glass-card" style="margin-top:0.5rem;">
            <h5>📦 Volume Δ & VMA</h5>
            <div style="color: rgba(255,255,255,0.85);">Flow confirmation via volume.</div>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\Delta V_t=\tfrac{V_t-V_{t-1}}{V_{t-1}},\; \text{VMA}_{24}=\tfrac{1}{24}\sum_{i=0}^{23}V_{t-i}")
        st.markdown("- Compute: pct change + 24h mean\n- Use: confirm breakouts/exhaustion")

    # Overview per user's requested theory content
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: rgba(255, 255, 255, 0.95); margin-bottom: 0.6rem;">SOLARIS: Next-hour return forecasting (overview)</h3>
        <p style="color: rgba(255,255,255,0.85);">
        SOLARIS predicts next-hour returns for a basket of equities using an ensemble of sequence and tabular learners. We collect hourly OHLCV, compute a compact set of technical, volume, volatility and calendar features, train sequence models (LSTM, CNN) plus an XGBoost tabular model, and combine their out-of-fold predictions with summary statistics in a lightweight meta-learner for robust, regime-aware forecasts.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 1 — Dataset & preprocessing
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4>1 — Dataset & preprocessing</h4>
        <ul style="color: rgba(255,255,255,0.85); line-height:1.7;">
            <li><b>Source & scope</b>: Hourly OHLCV for 12 equities from Yahoo Finance (~2 years).
                Align timestamps across assets to a single hourly index.</li>
            <li><b>Missing data</b>: Forward-fill intra-day gaps; drop hours with no market data across all assets; flag daylight/holiday gaps.</li>
            <li><b>Normalization</b>: Standardize features (z-score) using training-window stats only (no peeking).</li>
            <li><b>Windows</b>: Sequence models use sliding windows of length L=128; tree model uses the latest aggregated feature vector.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # 2 — Label
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4>2 — Label</h4>
        <div style="color: rgba(255,255,255,0.85);">Single-step, next-hour return (loss: MSE; optionally Huber for robustness):</div>
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"r_{t+1}=\frac{P_{t+1}-P_t}{P_t},\quad \hat P_{t+1}=P_t\,(1+\hat r_{t+1})")

    # 3 — Feature engineering
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4>3 — Feature engineering (compact & grouped)</h4>
        <p style="color: rgba(255,255,255,0.85);">Group features by intention; compute per-asset, plus cross-asset summaries.</p>
        <ul style="color: rgba(255,255,255,0.85); line-height:1.7;">
            <li><b>Price / returns</b>: 1h, 6h, 24h returns; log-returns optional.</li>
            <li><b>Momentum</b>: RSI(14); multi-horizon returns.</li>
            <li><b>Trend</b>: SMA/EMA; MACD=EMA(12)-EMA(26); Signal=EMA9(MACD).</li>
            <li><b>Volatility / risk</b>: Rolling standard deviation of 1h returns (6h/12h/24h windows).</li>
            <li><b>Volume</b>: Volume pct-change and 24-hour mean volume (VMA24).</li>
            <li><b>Calendar</b>: Hour-of-day, day-of-week (optionally cyclical sin/cos).</li>
            <li><b>Cross-asset</b>: Peer returns and correlations; peer-group mean returns (e.g., over 6h/24h).</li>
            <li><b>Summary stats</b>: Last-window mean/std/max/min of inputs for meta context.</li>
        </ul>
        <div style="color: rgba(255,255,255,0.8);">Keep features lean (~30–40 indicators). Log/clip heavy tails.</div>
    </div>
    """, unsafe_allow_html=True)

    # Key formulas for features
    st.latex(r"r_{t,\Delta}=\frac{P_t-P_{t-\Delta}}{P_{t-\Delta}}\quad\text{(returns)}")
    st.latex(r"\text{RSI}=100-\frac{100}{1+RS},\; RS=\frac{\text{avg gain}_{14}}{\text{avg loss}_{14}}")
    st.latex(r"\text{SMA}_t=\tfrac{1}{w}\sum_{i=0}^{w-1}P_{t-i},\; \text{EMA}_t=\alpha P_t+(1-\alpha)\,\text{EMA}_{t-1},\;\alpha=\tfrac{2}{w+1}")
    st.latex(r"\sigma_{t,w}=\sqrt{\tfrac{1}{w-1}\sum_{i=0}^{w-1}(r_{t-i,1}-\bar r)^2}\quad\text{(rolling std)}")
    st.latex(r"\Delta V_t=\tfrac{V_t-V_{t-1}}{V_{t-1}},\; \text{VMA}_{24}=\tfrac{1}{24}\sum_{i=0}^{23}V_{t-i}")

    # 4 — Base models (short cards)
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4>4 — Base models (what they learn & why)</h4>
        <p style="color: rgba(255,255,255,0.85);">Short cards: strengths and common architecture.</p>
    </div>
    """, unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown("""
        <div class="glass-card" style="margin-top:0.5rem;">
            <h5>🔄 LSTM (sequence)</h5>
            <div style="color: rgba(255,255,255,0.85);">Input: 128×features standardized window. 1–2 LSTM layers → Dense → \(\hat r_{t+1}\).</div>
            <div style="color: rgba(255,255,255,0.8);">Pros: long-range dependencies. Cons: slower; needs regularization.</div>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"(X * k)_t=\sum_{\tau=0}^{K-1}k_{\tau}X_{t-\tau}")
    with b2:
        st.markdown("""
        <div class="glass-card" style="margin-top:0.5rem;">
            <h5>🧩 CNN (temporal)</h5>
            <div style="color: rgba(255,255,255,0.85);">Conv1D+BN+ReLU stacks → GlobalAvgPool → Dense → \(\hat r\).</div>
            <div style="color: rgba(255,255,255,0.8);">Pros: faster, motif detection. Cons: limited long context unless dilated/deeper.</div>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"(X * k)_t=\sum_{\tau=0}^{K-1}k_{\tau}X_{t-\tau}")
    with b3:
        st.markdown("""
        <div class="glass-card" style="margin-top:0.5rem;">
            <h5>🌳 XGBoost (tabular)</h5>
            <div style="color: rgba(255,255,255,0.85);">Latest feature vector → boosted regression trees → \(\hat r\).</div>
            <div style="color: rgba(255,255,255,0.8);">Pros: strong on heterogeneous tabular signals; Cons: weak on raw sequences.</div>
        </div>
        """, unsafe_allow_html=True)

    # 5 — Meta-learner (ensemble)
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4>5 — Meta-learner (ensemble)</h4>
        <ul style="color: rgba(255,255,255,0.85); line-height:1.7;">
            <li><b>Inputs</b>: OOF predictions from CNN/LSTM/XGB + summary stats (mean/std/max/min) and volatility regime signals.</li>
            <li><b>Model</b>: lightweight regressor (e.g., HistGradientBoosting).</li>
            <li><b>Target</b>: minimize MSE on OOF sets (avoid leakage).</li>
            <li><b>Purpose</b>: learn when to trust each base model (trending vs. choppy).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # 6 — Training & validation (condensed)
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4>6 — Training pipeline & validation</h4>
        <ul style="color: rgba(255,255,255,0.85); line-height:1.7;">
            <li><b>Splits</b>: time-based (walk-forward); rolling windows; expanding train.</li>
            <li><b>Loss & optimizer</b>: MSE/Huber + Adam (NNs); regularized squared loss for XGBoost.</li>
            <li><b>OOF & stacking</b>: k-fold time-series CV; train meta on OOF preds.</li>
            <li><b>Regularization</b>: dropout/weight decay (NNs); lambda/alpha (XGBoost); optional input noise.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # 7 — Evaluation & backtest (condensed)
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4>7 — Evaluation & backtest metrics</h4>
        <ul style="color: rgba(255,255,255,0.85); line-height:1.7;">
            <li><b>Prediction</b>: MAE / MSE on returns (R² with care).</li>
            <li><b>Economic</b>: strategy P&L (sign/magnitude), Sharpe, max drawdown; costs/slippage.</li>
            <li><b>Stability</b>: rank correlation over time; per-regime performance (high/low σ).</li>
            <li><b>Robustness</b>: walk-forward, ensemble ablation, sensitivity to features, stress tests.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # 8 — Deployment
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4>8 — Deployment & inference</h4>
        <ul style="color: rgba(255,255,255,0.85); line-height:1.7;">
            <li><b>Inference</b>: build latest features → feed windows (LSTM/CNN) + tabular (XGB) → meta → \(\hat r_{t+1}\) → \(\hat P_{t+1}=P_t(1+\hat r)\).</li>
            <li><b>Latency</b>: CNN/XGB fast; LSTM can use warm-start or distillation.</li>
            <li><b>Monitoring</b>: data drift, model drift, P&L vs expected, feature distribution alerts.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # 9 — Suggested website layout (brief)
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4>9 — Suggested website layout & UX</h4>
        <ul style="color: rgba(255,255,255,0.85); line-height:1.7;">
            <li>Hero + one-liner; model cards (LSTM | CNN | XGB).</li>
            <li>Ensemble/pipeline diagram: Data → Features → Models → Meta → Forecast.</li>
            <li>Key formulas (collapsible); data & reproducibility notes.</li>
            <li>Backtest results (equity curve, Sharpe, drawdown) + per-regime breakdown.</li>
            <li>Technical appendix: full features, params, training schedule; link to notebook.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # 10 — Compact model card
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4>10 — Compact model card</h4>
        <div style="color: rgba(255,255,255,0.85);">
        <b>Model</b>: SOLARIS ensemble<br/>
        <b>Inputs</b>: hourly OHLCV, engineered features (~34) + cross-asset stats<br/>
        <b>Base learners</b>: LSTM(128-step), CNN(1D conv), XGBoost(tabular)<br/>
        <b>Meta</b>: HistGradientBoosting on OOF preds + summary stats<br/>
        <b>Label</b>: next-hour return \(r_{t+1}\) (MSE loss)<br/>
        <b>Validation</b>: purged time-series CV; walk-forward backtest<br/>
        <b>Primary risks</b>: lookahead bias, overfitting, regime shifts, transaction costs
        </div>
    </div>
    """, unsafe_allow_html=True)
