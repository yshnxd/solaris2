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
    page_title="StockWise: ML Based Stock Price Prediction",
    page_icon="📈",
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
        max-width: 100% !important;
        width: 100% !important;
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
    
    /* Professional Background */
    body {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Glassmorphism Effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 1.25rem;
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
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
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
    
    /* Fix chart and container width issues */
    .stPlotlyChart {
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
    }
    
    .stPlotlyChart > div {
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
    }
    
    /* Fix main container width and remove constraints (robust selectors) */
    .stApp [data-testid="block-container"] {
        padding-left: 0 !important;
        padding-right: 0 !important;
        max-width: 100% !important;
        width: 100% !important;
    }
    
    /* Override any width constraints from Streamlit root containers */
    .stApp,
    .stApp > div,
    .stApp [data-testid="stAppViewContainer"],
    .stApp [data-testid="stMain"] {
        max-width: 100vw !important;
        width: 100vw !important;
    }
    html, body { width: 100vw; overflow-x: hidden; }
    
    /* Fix any column width issues */
    [data-testid="stHorizontalBlock"],
    [data-testid="stVerticalBlock"],
    [data-testid="column"],
    [data-testid="stSidebarNav"] {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    /* Fix plotly chart container */
    [data-testid="stPlotlyChart"] {
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
    }

    /* Ensure custom containers stretch */
    .chart-container,
    .prediction-container,
    .glass-card {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Fix column spacing */
    .stColumns {
        gap: 1rem;
    }
    
    
    /* Fix glass card and prediction card sizing */
    .glass-card {
        min-height: auto;
        width: 100% !important;
        max-width: 100% !important;
        margin: 0.5rem 0;
    }
    
    .prediction-card {
        min-height: 200px;
        width: 100% !important;
        max-width: 100% !important;
    }
    
    /* Fix column containers */
    .stColumns {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    /* Fix metric containers */
    .stMetric {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    /* Simple separation between chart and predictions */
    .chart-container {
        margin-bottom: 2rem;
    }
    
    .prediction-container {
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Navigation and App Structure
st.markdown("""
<h1 class="main-header">📈 StockWise</h1>
""", unsafe_allow_html=True)

st.markdown("""
<h2 class="sub-header">Hybrid Machine Learning Stock Price Prediction System</h2>
""", unsafe_allow_html=True)

# Initialize session state
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Dashboard"

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

# Store the button states
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

# Main Navigation Tabs
tab1, tab2, tab3 = st.tabs(["🏠 Dashboard", "📚 Prediction History", "📚 Theory"])

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
        if df.empty:
            return pd.DataFrame(columns=show_cols)
    except (pd.errors.EmptyDataError, FileNotFoundError, Exception) as e:
        print(f"Error reading CSV: {e}")
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
    total_tickers = len(ticker_groups)
    for j, (ticker, indices) in enumerate(ticker_groups.items(), start=1):
        yf_ticker = ticker
        print(f"Processing ticker {j}/{total_tickers}: {ticker}")
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

# Global button handlers (work from any tab)
if evaluate_results:
    if os.path.exists(PREDICTION_HISTORY_CSV):
        with st.spinner("📋 Evaluating prediction history..."):
            try:
                updated_df = batch_evaluate_predictions_csv(PREDICTION_HISTORY_CSV)
                st.success("✅ Prediction history updated with actual prices!")
                st.rerun()  # Refresh the page to show updated data
            except Exception as e:
                st.error(f"❌ Error evaluating predictions: {str(e)}")
    else:
        st.warning("📋 No prediction history found to evaluate.")

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
    
    # Quick Stats Row
    col1, col2, col3, col4 = st.columns(4)
    
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
    
    with col4:
        st.markdown("""
        <div class="glass-card metric-card">
            <div style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">⚡</div>
                <div style="font-size: 1.5rem; color: #00ff88; font-weight: 900;">Live</div>
                <div style="color: rgba(255, 255, 255, 0.8); font-weight: 600;">Real-Time</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    

    # Recent Activity
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
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem; background: rgba(255, 255, 255, 0.1); border-radius: 8px; margin: 0.5rem 0;">
                        <span style="color: rgba(255, 255, 255, 0.9); font-weight: 600;">{row.get('ticker', 'N/A')}</span>
                        <span style="color: {signal_color}; font-weight: 700;">{row.get('signal', 'N/A')}</span>
                        <span style="color: rgba(255, 255, 255, 0.7);">${row.get('predicted_price', 0):.2f}</span>
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

    # Real-Time Market Data and Charts (only in Dashboard)
    st.markdown('<h2 class="section-header">📊 Live Stock Market Graph</h2>', unsafe_allow_html=True)
    
    # Fetch real-time data
main_ticker_data = fetch_stock_data(selected_ticker, period=f"{days_history}d", interval="60m")

if main_ticker_data is None or main_ticker_data.empty or len(main_ticker_data) < 128:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 3rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📈</div>
            <h3 style="color: rgba(255, 255, 255, 0.9); margin-bottom: 1rem;">No Chart Data Available</h3>
            <p style="color: rgba(255, 255, 255, 0.7);">
                Please select a valid stock ticker and ensure sufficient data is available.
            </p>
        </div>
        """, unsafe_allow_html=True)
else:
    latest_data = main_ticker_data.iloc[-1]
    current_price = latest_data['Close']
    current_time = latest_data.name
    
    # Enhanced Metrics Display
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
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{change_icon}</div>
                <div style="font-size: 1.5rem; color: {change_color}; font-weight: 900;">{change_formatted}</div>
                <div style="font-size: 1.1rem; color: {change_color}; font-weight: 600;">{change_pct_formatted}</div>
                <div style="color: rgba(255, 255, 255, 0.8); font-weight: 600;">24h Change</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
        with col3:
            volume = latest_data['Volume']
            volume_formatted = f"{volume:,.0f}"
            st.markdown(f"""
            <div class="glass-card metric-card">
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>
                    <div style="font-size: 1.5rem; color: #667eea; font-weight: 900;">{volume_formatted}</div>
                    <div style="color: rgba(255, 255, 255, 0.8); font-weight: 600;">Volume</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Chart and predictions side-by-side layout
        left_col, right_col = st.columns([2, 1], gap="medium")

        with left_col:
            st.subheader("📈 Interactive Price Chart")
            
            fig = go.Figure()
            
            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=main_ticker_data.index,
                open=main_ticker_data['Open'],
                high=main_ticker_data['High'],
                low=main_ticker_data['Low'],
                close=main_ticker_data['Close'],
                name="Price",
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff6b6b'
            ))
            
            # Add moving averages based on user selection
            if "SMA 5" in overlay_options:
                sma_5 = main_ticker_data['Close'].rolling(window=5).mean()
                fig.add_trace(go.Scatter(x=main_ticker_data.index, y=sma_5, name='SMA 5', line=dict(color='#667eea', width=2)))
            
            if "SMA 10" in overlay_options:
                sma_10 = main_ticker_data['Close'].rolling(window=10).mean()
                fig.add_trace(go.Scatter(x=main_ticker_data.index, y=sma_10, name='SMA 10', line=dict(color='#764ba2', width=2)))
            
            if "SMA 20" in overlay_options:
                sma_20 = main_ticker_data['Close'].rolling(window=20).mean()
                fig.add_trace(go.Scatter(x=main_ticker_data.index, y=sma_20, name='SMA 20', line=dict(color='#f093fb', width=2)))
            
            if "EMA 5" in overlay_options:
                ema_5 = main_ticker_data['Close'].ewm(span=5).mean()
                fig.add_trace(go.Scatter(x=main_ticker_data.index, y=ema_5, name='EMA 5', line=dict(color='#00ff88', width=2, dash='dash')))
            
            if "EMA 10" in overlay_options:
                ema_10 = main_ticker_data['Close'].ewm(span=10).mean()
                fig.add_trace(go.Scatter(x=main_ticker_data.index, y=ema_10, name='EMA 10', line=dict(color='#ff6b6b', width=2, dash='dash')))
            
            if "EMA 20" in overlay_options:
                ema_20 = main_ticker_data['Close'].ewm(span=20).mean()
                fig.add_trace(go.Scatter(x=main_ticker_data.index, y=ema_20, name='EMA 20', line=dict(color='#667eea', width=2, dash='dash')))
            
            fig.update_layout(
                title=f"{selected_ticker} Price Chart",
                xaxis_title="Time",
                yaxis_title="Price ($)",
                template="plotly_dark",
                height=600,
                width=None,
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                margin=dict(l=0, r=0, t=30, b=0),
            )
            
            st.plotly_chart(fig, use_container_width=True)

        with right_col:
            st.subheader("🎯 AI Predictions")
            
            prediction_triggered = False
            if run_prediction:
                prediction_triggered = True
            
            if st.button("🎯 Generate AI Prediction", key="main_prediction", help="Run AI prediction for selected stock") or prediction_triggered:
                with st.spinner("🤖 Running AI models..."):
                    try:
                        cross_data = fetch_cross_assets(main_ticker_data.index, days_history, selected_ticker)
                        features = create_features_for_app(selected_ticker, main_ticker_data, cross_data)
                        
                        if features is not None and len(features) >= SEQ_LEN:
                            predicted_price, pred_change_pct, votes, confidence = predict_with_models(
                                features, current_price, scaler, cnn_model, lstm_model, xgb_model, meta_model
                            )
                            
                            if predicted_price is not None:
                                st.subheader("🎯 Prediction Results")
                                
                                # Price metrics
                                price_col1, price_col2 = st.columns(2, gap="small")
                                
                                with price_col1:
                                    st.markdown(f"""
                                    <div class="metric-card" style="padding: 1rem; text-align: center;">
                                        <div style="font-size: 0.75rem; color: rgba(255, 255, 255, 0.8); margin-bottom: 0.5rem;">CURRENT</div>
                                        <div style="font-size: 1.5rem; color: #00ff88; font-weight: 900;">${current_price:.2f}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with price_col2:
                                    st.markdown(f"""
                                    <div class="metric-card" style="padding: 1rem; text-align: center; background: rgba(102, 126, 234, 0.3); border: 2px solid #667eea;">
                                        <div style="font-size: 0.75rem; color: rgba(255, 255, 255, 0.9); margin-bottom: 0.5rem; font-weight: 600;">PREDICTED</div>
                                        <div style="font-size: 1.5rem; color: #667eea; font-weight: 900;">${predicted_price:.2f}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Change and confidence
                                stats_col1, stats_col2 = st.columns(2, gap="small")
                                
                                with stats_col1:
                                    st.markdown(f"""
                                    <div class="metric-card" style="padding: 1rem; text-align: center;">
                                        <div style="font-size: 0.75rem; color: rgba(255, 255, 255, 0.8); margin-bottom: 0.5rem;">CHANGE</div>
                                        <div style="font-size: 1.5rem; color: {'#00ff88' if pred_change_pct >= 0 else '#ff6b6b'}; font-weight: 900;">{pred_change_pct:+.2f}%</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with stats_col2:
                                    st.markdown(f"""
                                    <div class="metric-card" style="padding: 1rem; text-align: center;">
                                        <div style="font-size: 0.75rem; color: rgba(255, 255, 255, 0.8); margin-bottom: 0.5rem;">CONFIDENCE</div>
                                        <div style="font-size: 1.5rem; color: {'#00ff88' if confidence > 70 else '#ff6b6b' if confidence < 50 else '#667eea'}; font-weight: 900;">{confidence:.1f}%</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Model consensus
                                st.markdown('<h5 style="color: rgba(255, 255, 255, 0.9); margin: 1rem 0 0.5rem 0; text-align: center;">🤖 Model Consensus</h5>', unsafe_allow_html=True)
                                
                                vote_col1, vote_col2, vote_col3 = st.columns(3, gap="small")
                                
                                with vote_col1:
                                    vote_color = "#00ff88" if votes["CNN"] == "UP" else "#ff6b6b"
                                    st.markdown(f"""
                                    <div class="metric-card" style="padding: 0.75rem; text-align: center;">
                                        <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">🧩</div>
                                        <div style="font-size: 0.85rem; color: rgba(255, 255, 255, 0.8); margin-bottom: 0.25rem;">CNN</div>
                                        <div style="font-size: 1rem; color: {vote_color}; font-weight: 700;">{votes["CNN"]}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with vote_col2:
                                    vote_color = "#00ff88" if votes["LSTM"] == "UP" else "#ff6b6b"
                                    st.markdown(f"""
                                    <div class="metric-card" style="padding: 0.75rem; text-align: center;">
                                        <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">🔄</div>
                                        <div style="font-size: 0.85rem; color: rgba(255, 255, 255, 0.8); margin-bottom: 0.25rem;">LSTM</div>
                                        <div style="font-size: 1rem; color: {vote_color}; font-weight: 700;">{votes["LSTM"]}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with vote_col3:
                                    vote_color = "#00ff88" if votes["XGBoost"] == "UP" else "#ff6b6b"
                                    st.markdown(f"""
                                    <div class="metric-card" style="padding: 0.75rem; text-align: center;">
                                        <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">🌳</div>
                                        <div style="font-size: 0.85rem; color: rgba(255, 255, 255, 0.8); margin-bottom: 0.25rem;">XGBoost</div>
                                        <div style="font-size: 1rem; color: {vote_color}; font-weight: 700;">{votes["XGBoost"]}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Save prediction to history
                                target_time = datetime.now() + timedelta(hours=1)
                                prediction_record = {
                                    'timestamp': datetime.now(),
                                    'ticker': selected_ticker,
                                    'current_price': current_price,
                                    'predicted_price': predicted_price,
                                    'target_time': target_time,
                                    'actual_price': None,
                                    'error_pct': None,
                                    'error_abs': None,
                                    'confidence': confidence,
                                    'signal': 'HOLD'
                                }
                                
                                ensure_csv_has_header(PREDICTION_HISTORY_CSV, show_cols)
                                df = pd.read_csv(PREDICTION_HISTORY_CSV)
                                df = pd.concat([df, pd.DataFrame([prediction_record])], ignore_index=True)
                                df.to_csv(PREDICTION_HISTORY_CSV, index=False)
                                
                                st.success("✅ Prediction saved to history!")
                                
                            else:
                                st.error("❌ Failed to generate prediction. Please check your data.")
                        else:
                            st.error("❌ Insufficient data for prediction. Need at least 128 data points.")
                            
                    except Exception as e:
                        st.error(f"❌ Error generating prediction: {str(e)}")
        
        
        # Handle sidebar watchlist button
        if run_watchlist:
            st.markdown('<h3 style="color: rgba(255, 255, 255, 0.9); margin: 2rem 0 1rem 0;">📊 Watchlist Analysis</h3>', unsafe_allow_html=True)
            
            # Default watchlist stocks
            watchlist_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
            
            st.markdown("""
            <div class="glass-card" style="margin: 1rem 0; padding: 2rem;">
                <h4 style="color: rgba(255, 255, 255, 0.9); margin-bottom: 1rem;">📈 Running Predictions for Watchlist Stocks</h4>
                <p style="color: rgba(255, 255, 255, 0.7); margin-bottom: 1rem;">
                    Analyzing {len(watchlist_stocks)} popular stocks with AI models...
            </p>
        </div>
        """, unsafe_allow_html=True)
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            watchlist_results = []
            
            for i, stock in enumerate(watchlist_stocks):
                status_text.text(f"Analyzing {stock}...")
                
                try:
                    # Fetch data for this stock
                    stock_data = fetch_stock_data(stock, period=f"{days_history}d", interval="60m")
                    
                    if stock_data is not None and len(stock_data) >= SEQ_LEN:
                        latest_data = stock_data.iloc[-1]
                        current_price = latest_data['Close']
                        
                        # Fetch cross-asset data
                        cross_data = fetch_cross_assets(stock_data.index, days_history, stock)
                        
                        # Create features
                        features = create_features_for_app(stock, stock_data, cross_data)
                        
                        if features is not None and len(features) >= SEQ_LEN:
                            # Make prediction
                            predicted_price, pred_change_pct, votes, confidence = predict_with_models(
                                features, current_price, scaler, cnn_model, lstm_model, xgb_model, meta_model
                            )
                            
                            if predicted_price is not None:
                                watchlist_results.append({
                                    'ticker': stock,
                                    'current_price': current_price,
                                    'predicted_price': predicted_price,
                                    'change_pct': pred_change_pct,
                                    'confidence': confidence,
                                    'votes': votes
                                })
                
                except Exception as e:
                    st.warning(f"⚠️ Could not analyze {stock}: {str(e)}")
                
                # Update progress
                progress_bar.progress((i + 1) / len(watchlist_stocks))
                time.sleep(0.5)  # Small delay for better UX
            
            status_text.text("✅ Watchlist analysis complete!")
            
            # Display results
            if watchlist_results:
                st.markdown('<h4 style="color: rgba(255, 255, 255, 0.9); margin: 2rem 0 1rem 0;">📊 Watchlist Results</h4>', unsafe_allow_html=True)
                
                # Sort by confidence
                watchlist_results.sort(key=lambda x: x['confidence'], reverse=True)
                
                for result in watchlist_results:
                    change_color = "#00ff88" if result['change_pct'] >= 0 else "#ff6b6b"
                    confidence_color = "#00ff88" if result['confidence'] > 70 else "#ff6b6b" if result['confidence'] < 50 else "#667eea"
                    
                    st.markdown(f"""
                    <div class="glass-card" style="margin: 1rem 0; padding: 1.5rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <div style="font-size: 1.3rem; color: rgba(255, 255, 255, 0.9); font-weight: 700;">{result['ticker']}</div>
                            <div style="font-size: 1rem; color: {confidence_color}; font-weight: 700; padding: 0.5rem 1rem; background: rgba(255, 255, 255, 0.1); border-radius: 20px;">{result['confidence']:.1f}% Confidence</div>
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 1rem;">
                            <div>
                                <div style="color: rgba(255, 255, 255, 0.7); font-size: 0.9rem;">Current Price</div>
                                <div style="color: #00ff88; font-weight: 600; font-size: 1.1rem;">${result['current_price']:.2f}</div>
                            </div>
                            <div style="background: rgba(102, 126, 234, 0.2); padding: 0.5rem; border-radius: 8px; border: 1px solid #667eea;">
                                <div style="color: rgba(255, 255, 255, 0.7); font-size: 0.9rem;">PREDICTED PRICE</div>
                                <div style="color: #667eea; font-weight: 700; font-size: 1.1rem;">${result['predicted_price']:.2f}</div>
                            </div>
                            <div>
                                <div style="color: rgba(255, 255, 255, 0.7); font-size: 0.9rem;">Expected Change</div>
                                <div style="color: {change_color}; font-weight: 600; font-size: 1.1rem;">{result['change_pct']:+.2f}%</div>
                            </div>
                            <div>
                                <div style="color: rgba(255, 255, 255, 0.7); font-size: 0.9rem;">Model Consensus</div>
                                <div style="color: rgba(255, 255, 255, 0.8); font-weight: 600; font-size: 0.9rem;">
                                    {result['votes']['CNN']}/{result['votes']['LSTM']}/{result['votes']['XGBoost']}
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Summary statistics
                avg_confidence = sum(r['confidence'] for r in watchlist_results) / len(watchlist_results)
                bullish_count = sum(1 for r in watchlist_results if r['change_pct'] > 0)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Average Confidence", f"{avg_confidence:.1f}%")
                
                with col2:
                    st.metric("Bullish Predictions", f"{bullish_count}/{len(watchlist_results)}")
                
                with col3:
                    st.metric("Stocks Analyzed", len(watchlist_results))
                
            else:
                st.warning("⚠️ No predictions could be generated for the watchlist stocks.")
        
        # Technical Indicators Section
        if show_rsi or show_macd:
            st.markdown('<h3 style="color: rgba(255, 255, 255, 0.9); margin: 2rem 0 1rem 0;">📊 Technical Indicators</h3>', unsafe_allow_html=True)
            
            if show_rsi:
                # RSI Chart
                rsi_values = ta.momentum.RSIIndicator(main_ticker_data['Close'], window=14).rsi()
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=main_ticker_data.index, y=rsi_values, name='RSI', line=dict(color='#667eea')))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
                fig_rsi.update_layout(
                    title="RSI (14)",
                    yaxis_title="RSI",
                    template="plotly_dark",
                    height=300,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            if show_macd:
                # MACD Chart
                macd_indicator = ta.trend.MACD(main_ticker_data['Close'])
                macd_line = macd_indicator.macd()
                signal_line = macd_indicator.macd_signal()
                histogram = macd_indicator.macd_diff()
                
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=main_ticker_data.index, y=macd_line, name='MACD', line=dict(color='#00ff88')))
                fig_macd.add_trace(go.Scatter(x=main_ticker_data.index, y=signal_line, name='Signal', line=dict(color='#ff6b6b')))
                fig_macd.add_trace(go.Bar(x=main_ticker_data.index, y=histogram, name='Histogram', marker_color='#667eea'))
                fig_macd.update_layout(
                    title="MACD",
                    yaxis_title="MACD",
                    template="plotly_dark",
                    height=300,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig_macd, use_container_width=True)

# Prediction History Tab
with tab2:
    st.markdown('<h2 class="section-header">📚 Prediction History</h2>', unsafe_allow_html=True)
    
    # Load and display prediction history
    if os.path.exists(PREDICTION_HISTORY_CSV):
        try:
            history_df = pd.read_csv(PREDICTION_HISTORY_CSV)
            if not history_df.empty:
                # Convert timestamp columns
                for col in ['timestamp', 'target_time']:
                    if col in history_df.columns:
                        history_df[col] = pd.to_datetime(history_df[col], errors='coerce')
                
                # Sort by timestamp
                history_df = history_df.sort_values('timestamp', ascending=False)
                
                # Display summary metrics - focus on prediction accuracy
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_predictions = len(history_df)
                    st.markdown(f"""
                    <div class="glass-card metric-card">
                        <div style="text-align: center;">
                            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>
                            <div style="font-size: 1.5rem; color: #667eea; font-weight: 900;">{total_predictions}</div>
                            <div style="color: rgba(255, 255, 255, 0.8); font-weight: 600;">Total Predictions</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Calculate average percentage error
                    completed_predictions = history_df[history_df['actual_price'].notna()]
                    if len(completed_predictions) > 0:
                        avg_error_pct = abs(completed_predictions['error_pct']).mean()
                    else:
                        avg_error_pct = 0
                    st.markdown(f"""
                    <div class="glass-card metric-card">
                        <div style="text-align: center;">
                            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>
                            <div style="font-size: 1.5rem; color: #ff6b6b; font-weight: 900;">{avg_error_pct:.2f}%</div>
                            <div style="color: rgba(255, 255, 255, 0.8); font-weight: 600;">Avg Error %</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    # Count completed predictions
                    completed_count = len(history_df[history_df['actual_price'].notna()])
                    st.markdown(f"""
                    <div class="glass-card metric-card">
                        <div style="text-align: center;">
                            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">✅</div>
                            <div style="font-size: 1.5rem; color: #f093fb; font-weight: 900;">{completed_count}</div>
                            <div style="color: rgba(255, 255, 255, 0.8); font-weight: 600;">Completed</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    # Calculate average absolute error in dollars
                    completed_predictions = history_df[history_df['actual_price'].notna()]
                    if len(completed_predictions) > 0:
                        avg_abs_error = completed_predictions['error_abs'].mean()
                    else:
                        avg_abs_error = 0
                    st.markdown(f"""
                    <div class="glass-card metric-card">
                        <div style="text-align: center;">
                            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">💰</div>
                            <div style="font-size: 1.5rem; color: #f093fb; font-weight: 900;">${avg_abs_error:.2f}</div>
                            <div style="color: rgba(255, 255, 255, 0.8); font-weight: 600;">Avg Error $</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add average confidence metric
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col5:
                    avg_confidence = history_df['confidence'].mean() if 'confidence' in history_df.columns else 0
                    st.markdown(f"""
                    <div class="glass-card metric-card">
                        <div style="text-align: center;">
                            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🔮</div>
                            <div style="font-size: 1.5rem; color: #667eea; font-weight: 900;">{avg_confidence:.1f}%</div>
                            <div style="color: rgba(255, 255, 255, 0.8); font-weight: 600;">Avg Confidence</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Action buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔄 Refresh Actual Prices", help="Update actual prices for completed predictions"):
                        with st.spinner("Updating actual prices..."):
                            updated_df = batch_evaluate_predictions_csv(PREDICTION_HISTORY_CSV)
                            st.success("✅ Actual prices updated!")
                            st.rerun()
                
                with col2:
                    if st.button("📊 View Detailed Table", help="Show detailed prediction history"):
                        st.markdown('<h3 style="color: rgba(255, 255, 255, 0.9); margin: 2rem 0 1rem 0;">📋 Detailed Prediction History</h3>', unsafe_allow_html=True)
                        
                        # Format the dataframe for display
                        display_df = history_df.copy()
                        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                        display_df['target_time'] = display_df['target_time'].dt.strftime('%Y-%m-%d %H:%M')
                        
                        # Format numeric columns
                        for col in ['current_price', 'predicted_price', 'actual_price']:
                            if col in display_df.columns:
                                display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
                        
                        for col in ['error_pct', 'confidence']:
                            if col in display_df.columns:
                                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
                        
                        for col in ['error_abs']:
                            if col in display_df.columns:
                                display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
                        
                        st.dataframe(display_df, use_container_width=True)
                
                # Recent predictions display
                st.markdown('<h3 style="color: rgba(255, 255, 255, 0.9); margin: 2rem 0 1rem 0;">🕒 Recent Predictions</h3>', unsafe_allow_html=True)
                
                recent_predictions = history_df.head(10)
                
                for _, row in recent_predictions.iterrows():
                    confidence_color = "#00ff88" if row.get('confidence', 0) > 70 else "#ff6b6b" if row.get('confidence', 0) < 50 else "#667eea"
                    
                    st.markdown(f"""
                    <div class="glass-card" style="margin: 1rem 0; padding: 1.5rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <div style="font-size: 1.2rem; color: rgba(255, 255, 255, 0.9); font-weight: 700;">{row.get('ticker', 'N/A')}</div>
                            <div style="font-size: 1.1rem; color: {confidence_color}; font-weight: 700; padding: 0.5rem 1rem; background: rgba(255, 255, 255, 0.1); border-radius: 20px;">{row.get('confidence', 0):.1f}% Confidence</div>
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                            <div>
                                <div style="color: rgba(255, 255, 255, 0.7); font-size: 0.9rem;">Current Price</div>
                                <div style="color: #00ff88; font-weight: 600;">${row.get('current_price', 0):.2f}</div>
                            </div>
                            <div style="background: rgba(102, 126, 234, 0.2); padding: 0.5rem; border-radius: 8px; border: 1px solid #667eea;">
                                <div style="color: rgba(255, 255, 255, 0.7); font-size: 0.9rem;">PREDICTED PRICE</div>
                                <div style="color: #667eea; font-weight: 700; font-size: 1.1rem;">${row.get('predicted_price', 0):.2f}</div>
                            </div>
                            <div>
                                <div style="color: rgba(255, 255, 255, 0.7); font-size: 0.9rem;">Actual Price</div>
                                <div style="color: #f093fb; font-weight: 600;">{'N/A' if not pd.notna(row.get('actual_price')) else f"${row.get('actual_price', 0):.2f}"}</div>
                            </div>
                            <div>
                                <div style="color: rgba(255, 255, 255, 0.7); font-size: 0.9rem;">Error</div>
                                <div style="color: {confidence_color}; font-weight: 600;">{f"{row.get('error_pct', 0):.2f}%" if pd.notna(row.get('error_pct')) else 'N/A'}</div>
                            </div>
                        </div>
                        <div style="color: rgba(255, 255, 255, 0.6); font-size: 0.9rem;">
                            Predicted: {row.get('timestamp', 'N/A')} | Target: {row.get('target_time', 'N/A')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
            else:
                st.markdown("""
                <div class="glass-card" style="text-align: center; padding: 3rem;">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
                    <h3 style="color: rgba(255, 255, 255, 0.9); margin-bottom: 1rem;">No Prediction History</h3>
                    <p style="color: rgba(255, 255, 255, 0.7);">
                        Generate your first prediction in the Dashboard tab to see history here.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error loading prediction history: {str(e)}")
    else:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 3rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
            <h3 style="color: rgba(255, 255, 255, 0.9); margin-bottom: 1rem;">No Prediction History</h3>
            <p style="color: rgba(255, 255, 255, 0.7);">
                Generate your first prediction in the Dashboard tab to see history here.
            </p>
        </div>
        """, unsafe_allow_html=True)
# Theory Tab
with tab3:
    st.markdown('<h2 class="section-header">📚 Theory: Models and Indicators</h2>', unsafe_allow_html=True)

    # --- Models Section ---
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: rgba(255, 255, 255, 0.95); margin-bottom: 1rem; text-align:center;">🧠 Models Used</h3>
        <p style="color: rgba(255,255,255,0.85);">StockWise uses an ensemble of sequence models and tabular learners combined with a meta-learner.</p>
    </div>
    """, unsafe_allow_html=True)

    # LSTM
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4 style="margin-bottom:0.5rem;">🔄 LSTM (Long Short-Term Memory)</h4>
        <p style="color: rgba(255,255,255,0.85);">Captures long-range temporal dependencies in price sequences.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**Mathematical Formulation**")
    st.latex(r"""
    \begin{aligned}
    &\text{Input sequence: } X = (x_{t-L+1},\dots,x_t) \\
    &\text{LSTM cell: } \\
    &f_t = \sigma(W_f [h_{t-1}, x_t] + b_f),\quad i_t = \sigma(W_i [h_{t-1}, x_t] + b_i),\\
    &\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c),\quad c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t,\\
    &o_t = \sigma(W_o [h_{t-1}, x_t] + b_o),\quad h_t = o_t \odot \tanh(c_t) \\
    &\text{Output: } \hat{y}_{t+1} = W_y h_t + b_y
    \end{aligned}
    """)
    st.markdown("**Loss / Optimization**: Mean Squared Error (MSE), optimized via backpropagation through time (BPTT) and gradient descent (e.g., Adam).")
    st.markdown("**Pros**: Good for sequential patterns, regime shifts; **Cons**: Slower to train, can overfit without regularization.")
    st.markdown("**Why for stocks**: Captures momentum, mean-reversion, and intraday cycles across windows.")
    st.markdown("**Pipeline (pseudo)**")
    st.markdown("""
```text
window <- last 128 standardized feature vectors
h <- LSTM(window)
predicted_return <- dense(h)
predicted_price <- current_price * (1 + predicted_return)
```
""")

    # CNN
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4 style="margin-bottom:0.5rem;">🧩 CNN (Temporal Convolution)</h4>
        <p style="color: rgba(255,255,255,0.85);">Learns local temporal patterns and motifs in rolling windows.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**Mathematical Formulation**")
    st.latex(r"\text{For kernel } k: \quad (X * k)_t = \sum_{\tau=0}^{K-1} k_{\tau} \cdot X_{t-\tau}")
    st.markdown("**Loss / Optimization**: MSE minimized via stochastic gradient descent (Adam).")
    st.markdown("**Pros**: Fast inference; local feature extraction (spikes, breakouts). **Cons**: Limited long-horizon context without dilation or stacking.")
    st.markdown("**Why for stocks**: Detects short-term impulses, microstructure patterns.")
    st.markdown("**Pipeline (pseudo)**")
    st.markdown("""
```text
window <- last 128 standardized feature vectors
features <- Conv1D/ReLU/Pooling stacks
predicted_return <- dense(features)
predicted_price <- current_price * (1 + predicted_return)
```
""")

    # XGBoost
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4 style="margin-bottom:0.5rem;">🌳 XGBoost (Gradient-Boosted Trees)</h4>
        <p style="color: rgba(255,255,255,0.85);">Learns tabular, non-linear interactions from the most recent feature vector.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**Mathematical Formulation**")
    st.latex(r"\hat{y}(x) = \sum_{m=1}^{M} \gamma_m T_m(x), \quad \text{with } \gamma_m \text{ fit by gradient boosting}")
    st.markdown("**Loss / Optimization**: Regularized objective with shrinkage; stage-wise additive modeling minimizing gradient of loss.")
    st.markdown("**Pros**: Handles heterogeneous features, robustness; **Cons**: Less effective for raw sequences.")
    st.markdown("**Why for stocks**: Strong on engineered indicators, cross-asset features, calendar effects.")
    st.markdown("**Pipeline (pseudo)**")
    st.markdown("""
```text
x_tab <- last standardized feature vector
predicted_return <- XGBoost(x_tab)
predicted_price <- current_price * (1 + predicted_return)
```
""")

    # Meta-learner
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4 style="margin-bottom:0.5rem;">🤝 Meta-Learner (Ensemble)</h4>
        <p style="color: rgba(255,255,255,0.85);">Combines CNN/LSTM/XGBoost predictions and simple summary stats.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**Formulation**")
    st.latex(r"\hat{r}_{meta} = g\big([\hat{r}_{cnn}, \hat{r}_{lstm}, \hat{r}_{xgb}, \mu, \sigma, \max, \min]\big)")
    st.markdown("Where g is a small model (e.g., linear or shallow net). Final price is \(\hat{P}_{t+1}=P_t (1+\hat{r}_{meta})\).")
    st.markdown("**Why**: Blends diverse biases/variances → improved stability.")

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
        <ul>
            <li><strong>SMA(w)</strong>: \(\text{SMA}_t = \frac{1}{w} \sum_{i=0}^{w-1} P_{t-i}\) — smooths noise, highlights direction.</li>
            <li><strong>EMA(w)</strong>: \(\text{EMA}_t = \alpha P_t + (1-\alpha)\,\text{EMA}_{t-1}\), \(\alpha=\frac{2}{w+1}\).</li>
            <li><strong>MACD</strong>: \(\text{MACD}_t = \text{EMA}_{12}(P)_t - \text{EMA}_{26}(P)_t\).</li>
            <li><strong>Signal</strong>: \(\text{Signal}_t = \text{EMA}_{9}(\text{MACD})_t\).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Momentum
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4>🏃 Momentum</h4>
        <ul>
            <li><strong>Returns</strong>: \(r_{t,\Delta} = \frac{P_t - P_{t-\Delta}}{P_{t-\Delta}}\) for \(\Delta\in\{1,3,6,12,24\}\,\text{hours}\).</li>
            <li><strong>RSI(14)</strong>: \(\text{RSI} = 100 - \frac{100}{1 + RS}\), \(RS=\frac{\text{avg gain}_{14}}{\text{avg loss}_{14}}\).</li>
            <li><strong>Cross-asset 1h returns</strong>: same return formula using peer tickers (AAPL, MSFT, ...).</li>
        </ul>
        <p style="color: rgba(255,255,255,0.8);">Momentum captures trend persistence; RSI flags overbought/oversold states.</p>
    </div>
    """, unsafe_allow_html=True)

    # Volatility
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4>🌪️ Volatility</h4>
        <ul>
            <li><strong>Rolling Std</strong>: \(\sigma_{t,w} = \sqrt{\frac{1}{w-1} \sum_{i=0}^{w-1} (r_{t-i,1}-\bar{r})^2}\) for windows 6h, 12h, 24h.</li>
        </ul>
        <p style="color: rgba(255,255,255,0.8);">Higher volatility widens expected price distribution; important for risk-adjusted signals.</p>
    </div>
    """, unsafe_allow_html=True)

    # Volume
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4>📦 Volume</h4>
        <ul>
            <li><strong>Volume change (1h)</strong>: \(\Delta V_t = \frac{V_t - V_{t-1}}{V_{t-1}}\).</li>
            <li><strong>Volume MA(24h)</strong>: \(\text{VMA}_{24} = \frac{1}{24} \sum_{i=0}^{23} V_{t-i}\).</li>
        </ul>
        <p style="color: rgba(255,255,255,0.8);">Volume confirms moves; surges can precede breakouts or exhaustion.</p>
    </div>
    """, unsafe_allow_html=True)

    # Calendar and Price
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4>🗓️ Calendar & Price</h4>
        <ul>
            <li><strong>Hour of day</strong>: captures intraday seasonality (open/close effects).</li>
            <li><strong>Day of week</strong>: weekday effects (e.g., Monday momentum).</li>
            <li><strong>Price</strong>: level feature that interacts with indicators (e.g., distance from averages).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Why helpful
    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <h4>💡 Why these help forecasting</h4>
        <ul>
            <li><strong>MAs</strong> smooth noise and reveal underlying direction.</li>
            <li><strong>RSI</strong> highlights stretched conditions that often mean-revert.</li>
            <li><strong>MACD</strong> detects trend changes via moving-average crossovers.</li>
            <li><strong>Volatility</strong> modulates confidence and expected move size.</li>
            <li><strong>Volume</strong> validates price moves; abnormal activity can signal regime shifts.</li>
            <li><strong>Cross-asset returns</strong> exploit correlation and leadership effects across related stocks.</li>
            <li><strong>Calendar</strong> encodes systematic intraday/weekly patterns.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
