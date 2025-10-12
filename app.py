import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import pickle
import random
import warnings
warnings.filterwarnings("ignore")

# Deterministic random seed for reproducibility
np.random.seed(42)
random.seed(42)

st.set_page_config(
    page_title="📈 StockWise: Hybrid AI Stock Prediction System",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# ---- Load Models and Scaler ----
@st.cache_resource
def load_models_and_scaler():
    with open("cnn_model.pkl", "rb") as f:
        cnn_model = pickle.load(f)
    with open("lstm_model.pkl", "rb") as f:
        lstm_model = pickle.load(f)
    with open("xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("meta_learner.pkl", "rb") as f:
        meta_learner = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return cnn_model, lstm_model, xgb_model, meta_learner, scaler

cnn_model, lstm_model, xgb_model, meta_learner, scaler = load_models_and_scaler()

# ---- Technical Indicator Functions ----
def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=1).mean()
    ma_down = down.rolling(window=period, min_periods=1).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_sma(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def compute_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def compute_lag_returns(series, lags):
    returns = []
    for lag in lags:
        returns.append(series.pct_change(periods=lag))
    return returns

def compute_rolling_volatility(series, windows):
    vols = []
    for window in windows:
        vols.append(series.pct_change().rolling(window=window, min_periods=1).std())
    return vols

# ---- Feature Engineering ----
def prepare_features(df):
    # Lag returns
    lags = [1, 3, 6, 12, 24]
    lag_returns = compute_lag_returns(df['Close'], lags)
    for i, lag in enumerate(lags):
        df[f'return_lag_{lag}h'] = lag_returns[i]

    # Rolling volatility
    windows = [6, 12, 24]
    rolling_vols = compute_rolling_volatility(df['Close'], windows)
    for i, win in enumerate(windows):
        df[f'vol_{win}h'] = rolling_vols[i]

    # RSI(14)
    df['RSI_14'] = compute_rsi(df['Close'], 14)

    # MACD and MACD Signal
    macd, macd_signal = compute_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_signal'] = macd_signal

    # SMA/EMA
    for w in [5, 10, 20]:
        df[f'SMA_{w}'] = compute_sma(df['Close'], w)
        df[f'EMA_{w}'] = compute_ema(df['Close'], w)

    # Volume change
    df['vol_chg_1h'] = df['Volume'].pct_change(periods=1)
    df['vol_chg_6h'] = df['Volume'].pct_change(periods=6)

    # Day of week, hour of day
    df['day_of_week'] = df.index.dayofweek
    df['hour_of_day'] = df.index.hour

    # Drop incomplete rows
    df = df.dropna().copy()
    return df

# ---- Model Prediction Pipeline ----
def predict_next_hour(ticker="AAPL", horizon=1):
    try:
        # Fetch last 48h of hourly data
        end = datetime.utcnow()
        start = end - timedelta(hours=60)
        data = yf.download(
            tickers=ticker,
            interval="1h",
            start=start.strftime('%Y-%m-%d %H:%M:%S'),
            end=end.strftime('%Y-%m-%d %H:%M:%S'),
            progress=False
        )

        if data.empty or len(data) < 30:
            raise ValueError("Not enough data retrieved. Try another ticker or wait for more market hours.")

        df = data.tail(48).copy()
        df = prepare_features(df)
        latest_features = df.iloc[-1].copy()

        # Extract features for models (make sure order matches training)
        feature_names = [
            'return_lag_1h', 'return_lag_3h', 'return_lag_6h', 'return_lag_12h', 'return_lag_24h',
            'vol_6h', 'vol_12h', 'vol_24h',
            'RSI_14', 'MACD', 'MACD_signal',
            'SMA_5', 'SMA_10', 'SMA_20',
            'EMA_5', 'EMA_10', 'EMA_20',
            'vol_chg_1h', 'vol_chg_6h',
            'day_of_week', 'hour_of_day'
        ]
        X = latest_features[feature_names].values.reshape(1, -1)
        X_scaled = scaler.transform(X)

        # All models are .pkl: sklearn/xgboost compatible
        cnn_pred = cnn_model.predict(X_scaled)[0]
        lstm_pred = lstm_model.predict(X_scaled)[0]
        xgb_pred = xgb_model.predict(X_scaled)[0]

        base_preds = np.array([cnn_pred, lstm_pred, xgb_pred]).reshape(1, -1)
        meta_pred = meta_learner.predict(base_preds)[0]

        # Convert predicted return to price
        latest_price = df['Close'].iloc[-1]
        predicted_return = meta_pred
        predicted_price = latest_price * (1 + predicted_return)

        # Confidence: use meta-learner's std or base model agreement
        base_std = np.std([cnn_pred, lstm_pred, xgb_pred])
        confidence = 1 - min(base_std, 0.2) / 0.2  # scaled [0,1], lower disagreement = higher confidence

        intermediate = {
            "CNN": cnn_pred,
            "LSTM": lstm_pred,
            "XGBoost": xgb_pred
        }

        result = {
            "ticker": ticker,
            "latest_price": latest_price,
            "predicted_price": predicted_price,
            "predicted_return": predicted_return,
            "confidence": confidence,
            "df": df,
            "intermediate": intermediate
        }
        return result
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None

# ---- Sidebar ----
st.sidebar.title("⚙️ StockWise Controls")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
horizon = st.sidebar.selectbox("Forecast Horizon (hours)", options=[1], index=0)
predict_btn = st.sidebar.button("🔮 Predict")

st.title("📈 StockWise: Hybrid AI Stock Prediction System")
st.markdown("---")

if predict_btn or ticker:
    with st.spinner(f"Fetching data and running prediction for {ticker}..."):
        result = predict_next_hour(ticker=ticker, horizon=horizon)
    if result:
        latest_price = result["latest_price"]
        predicted_price = result["predicted_price"]
        predicted_return = result["predicted_return"]
        confidence = result["confidence"]
        df = result["df"]
        intermediate = result["intermediate"]

        col1, col2, col3 = st.columns([2,2,2])
        col1.metric("Current Price", f"${latest_price:,.2f}")
        col2.metric("Predicted Next-Hour Price", f"${predicted_price:,.2f}",
                    delta=f"{(predicted_price-latest_price):+.2f}")
        col3.metric("Predicted Return (%)", f"{predicted_return*100:+.2f}%",
                    delta=None)
        st.info(f"Model Confidence: {confidence*100:.1f}%", icon="🤖")
        st.markdown("---")

        # ---- Price Plot ----
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'],
            mode='lines+markers', name='Actual Price',
            line=dict(color='royalblue')
        ))
        fig.add_trace(go.Scatter(
            x=[df.index[-1] + pd.Timedelta(hours=1)], y=[predicted_price],
            mode='markers', marker=dict(color='orange', size=12, symbol='diamond'),
            name='Predicted Price'
        ))
        fig.update_layout(
            title=f"{ticker} Price (last 48h) & Forecast",
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=True,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ---- RSI Chart ----
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=df.index, y=df['RSI_14'],
            mode='lines', name='RSI(14)', line=dict(color='green')
        ))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="blue")
        fig_rsi.update_layout(
            title="RSI(14) Indicator",
            yaxis_title="RSI Value",
            template="plotly_white"
        )
        st.plotly_chart(fig_rsi, use_container_width=True)

        # ---- MACD Chart ----
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(
            x=df.index, y=df['MACD'], name='MACD', line=dict(color='purple')
        ))
        fig_macd.add_trace(go.Scatter(
            x=df.index, y=df['MACD_signal'], name='MACD Signal', line=dict(color='orange')
        ))
        fig_macd.update_layout(
            title="MACD & Signal Line",
            yaxis_title="Value",
            template="plotly_white"
        )
        st.plotly_chart(fig_macd, use_container_width=True)

        # ---- Intermediate Model Outputs ----
        with st.expander("Show Intermediate Model Outputs"):
            st.markdown(f"**CNN Prediction:** {intermediate['CNN']:+.5f}")
            st.markdown(f"**LSTM Prediction:** {intermediate['LSTM']:+.5f}")
            st.markdown(f"**XGBoost Prediction:** {intermediate['XGBoost']:+.5f}")

        st.markdown("---")
        # ---- Optional: Performance Metrics ----
        with st.expander("Model Performance & Stack Logic"):
            st.markdown("""
            **Validation Metrics (from test phase):**
            - MAE: *see training logs*
            - MAPE: *see training logs*
            - R²: *see training logs*

            **Ensemble Logic:**  
            The system uses three base models (CNN, LSTM, XGBoost) trained on technical indicators and price features.  
            Their predictions are combined by a meta-model (HistGradientBoostingRegressor) to provide a robust next-hour return forecast.
            """)
else:
    st.info("Enter a ticker and click 🔮 Predict to get started.", icon="🔎")

st.markdown("---")
st.caption("Powered by Streamlit, yfinance, scikit-learn, xgboost, and Plotly. © 2025 StockWise. 🚀")
