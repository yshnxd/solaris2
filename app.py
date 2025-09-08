import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler

# ========== CACHED LOADING FUNCTIONS ==========

@st.cache_data
def load_models():
    cnn_model = joblib.load("cnn_model.pkl")
    lstm_model = joblib.load("lstm_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    meta_learner = joblib.load("meta_learner.pkl")
    scaler = joblib.load("scaler.pkl")
    return cnn_model, lstm_model, xgb_model, meta_learner, scaler

@st.cache_data
def get_stock_data(ticker, n_days):
    """
    Download hourly OHLCV for ticker from Yahoo Finance for n_days.
    """
    interval = "60m"
    period = f"{int(n_days)}d"
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    df.dropna(inplace=True)
    df.index = df.index.tz_localize(None)
    return df

@st.cache_data
def get_indices():
    indices = ["^DJI", "SPY", "QQQ"]
    dfs = []
    for idx in indices:
        df = yf.download(idx, period="5d", interval="1d", progress=False)
        df = df.iloc[-1:]
        dfs.append(df)
    return dict(zip(indices, dfs))

@st.cache_data
def get_feature_importance(xgb_model, feature_names):
    importance = xgb_model.feature_importances_
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)
    return imp_df.head(10)

# ========== NOTEBOOK-STYLE FEATURE ENGINEERING ==========

def create_features(df, all_ticker_data, ticker):
    """
    Feature creation logic based on notebook for a single ticker.
    df: DataFrame of OHLCV for target ticker.
    all_ticker_data: dict of {ticker: DataFrame} for cross returns.
    """
    price_series = df["Close"].copy()
    feat_tmp = pd.DataFrame(index=df.index)

    # Lag returns
    for lag in [1, 3, 6, 12, 24]:
        feat_tmp[f"ret_{lag}h"] = price_series.pct_change(lag)

    # Rolling volatility
    for window in [6, 12, 24]:
        feat_tmp[f"vol_{window}h"] = price_series.pct_change().rolling(window).std()

    # RSI 14
    try:
        import ta
        feat_tmp["rsi_14"] = ta.momentum.RSIIndicator(price_series, window=14).rsi()
    except Exception:
        feat_tmp["rsi_14"] = np.nan

    # MACD
    try:
        macd = ta.trend.MACD(price_series)
        feat_tmp["macd"] = macd.macd()
        feat_tmp["macd_signal"] = macd.macd_signal()
    except Exception:
        feat_tmp["macd"] = np.nan
        feat_tmp["macd_signal"] = np.nan

    # Moving averages
    for w in [5, 10, 20]:
        feat_tmp[f"sma_{w}"] = price_series.rolling(w).mean()
        feat_tmp[f"ema_{w}"] = price_series.ewm(span=w, adjust=False).mean()

    # Volume features
    if "Volume" in df.columns:
        vol_series = df["Volume"].ffill()
        feat_tmp["vol_change_1h"] = vol_series.pct_change()
        feat_tmp["vol_ma_24h"] = vol_series.rolling(24).mean()

    # Cross-asset returns (use the 12 trained tickers if available)
    trained_assets = [
        "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "NVDA", "JPM",
        "JNJ", "XOM", "CAT", "BA", "META"
    ]
    for asset in trained_assets:
        if asset in all_ticker_data:
            asset_close = all_ticker_data[asset]["Close"].reindex(df.index).ffill()
            feat_tmp[f"{asset}_ret_1h"] = asset_close.pct_change()

    # Calendar features
    feat_tmp["hour"] = feat_tmp.index.hour
    feat_tmp["day_of_week"] = feat_tmp.index.dayofweek
    feat_tmp["datetime"] = feat_tmp.index
    feat_tmp["ticker"] = ticker

    # Drop rows with NaNs in features relevant to prediction (except datetime, ticker)
    drop_cols = [col for col in feat_tmp.columns if col not in ["datetime", "ticker"]]
    feat_tmp = feat_tmp.dropna(subset=drop_cols)
    return feat_tmp

def make_prediction(feature_row, cnn_model, lstm_model, xgb_model, meta_learner, scaler):
    """
    Given a single row of features, generate all model predictions.
    """
    # Drop non-feature cols
    feature_cols = [col for col in feature_row.index if col not in ["datetime", "ticker"]]
    X_tab = feature_row[feature_cols].values.reshape(1, -1)
    X_tab_scaled = scaler.transform(X_tab)
    # CNN/LSTM expect sequence, so use last 128 rows (or as many as available)
    # For demo: pad with zeros if not enough
    X_seq = np.zeros((1, 128, X_tab.shape[1]))
    if len(X_tab_scaled.shape) == 2 and X_tab_scaled.shape[0] == 1:
        X_seq[0, -1, :] = X_tab_scaled[0]
    else:
        seq_len = min(128, X_tab_scaled.shape[0])
        X_seq[0, -seq_len:, :] = X_tab_scaled[-seq_len:]

    # Predict
    pred_cnn = float(cnn_model.predict(X_seq)[0][0])
    pred_lstm = float(lstm_model.predict(X_seq)[0][0])
    pred_xgb = float(xgb_model.predict(X_tab_scaled)[0])
    # Meta features for ensemble
    meta_X = np.array([[pred_cnn, pred_lstm, pred_xgb,
                        np.mean([pred_cnn, pred_lstm, pred_xgb]),
                        np.std([pred_cnn, pred_lstm, pred_xgb]),
                        np.max([pred_cnn, pred_lstm, pred_xgb]),
                        np.min([pred_cnn, pred_lstm, pred_xgb])]])
    pred_meta = float(meta_learner.predict(meta_X)[0])
    # Confidence = std of base model predictions (lower std = higher confidence)
    confidence = 1.0 - np.clip(np.std([pred_cnn, pred_lstm, pred_xgb]), 0, 1)
    # Direction votes
    votes = {
        "CNN": "UP" if pred_cnn > 0 else ("DOWN" if pred_cnn < 0 else "HOLD"),
        "LSTM": "UP" if pred_lstm > 0 else ("DOWN" if pred_lstm < 0 else "HOLD"),
        "XGBoost": "UP" if pred_xgb > 0 else ("DOWN" if pred_xgb < 0 else "HOLD"),
    }
    return pred_meta, pred_cnn, pred_lstm, pred_xgb, confidence, votes

# Prediction log (in-memory for demo, replace with persistent storage for production)
if "prediction_log" not in st.session_state:
    st.session_state["prediction_log"] = []

# ========== STREAMLIT APP ==========

st.set_page_config(
    page_title="SOLARIS : Hourly Stock Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- TITLE & INTRO --
st.title("SOLARIS : A Machine Learning Based Hourly Stock Prediction Tool")
st.subheader("A ML Model combining Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM) networks, and XGBoost with a Meta Learner Ensemble for Short Term Stock Market Prediction")
st.markdown(
    """
    _This project explores the potential of ensemble deep learning models to identify short-term patterns in financial time series data for hourly price prediction. It is for educational and research purposes only._
    """
)

st.warning("**Disclaimer:** This tool is for research and educational purposes only. Do not use for trading or investment decisions.")

# -- SECTION 1: LIVE PREDICTION DASHBOARD --

st.header("🔮 Live Prediction Dashboard")

col_input, col_days, col_btn = st.columns([2, 1, 1])
with col_input:
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):", value="AAPL")
with col_days:
    n_days = st.text_input("Days of Historical Data to Display (90-729):", value="180")
try:
    n_days = max(90, min(729, int(n_days)))
except Exception:
    n_days = 180

with col_btn:
    predict_btn = st.button("🚀 Predict Next Hour")

# Load models once
cnn_model, lstm_model, xgb_model, meta_learner, scaler = load_models()

# Download all 12 trained tickers for cross-asset returns
trained_tickers = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "NVDA", "JPM",
    "JNJ", "XOM", "CAT", "BA", "META"
]
all_ticker_data = {}
for t in trained_tickers:
    try:
        all_ticker_data[t] = get_stock_data(t, n_days)
    except Exception:
        all_ticker_data[t] = None

# Download target ticker
try:
    df = get_stock_data(ticker, n_days)
    current_price = df["Close"].iloc[-1]
except Exception:
    st.error(f"Could not fetch data for {ticker}. Please check the ticker symbol.")
    df = None
    current_price = None

if df is not None and not df.empty:
    # -- Feature Creation --
    features_df = create_features(df, all_ticker_data, ticker)
    last_feat_row = features_df.iloc[-1] if not features_df.empty else None

    # -- Price Chart --
    st.subheader(f"Live Price Chart for {ticker}")
    fig = go.Figure()
    # Plot historical prices
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines", name="Price", line=dict(color="blue")
    ))
    # Mark "NOW"
    fig.add_vline(x=df.index[-1], line_dash="dash", line_color="gray", annotation_text="NOW")
    # If prediction is made, add it to chart
    pred_price = None
    if predict_btn and last_feat_row is not None:
        pred_ret, pred_cnn, pred_lstm, pred_xgb, confidence, votes = make_prediction(
            last_feat_row, cnn_model, lstm_model, xgb_model, meta_learner, scaler)
        pred_price = current_price * (1 + pred_ret)
        # Add predicted price
        color = "green" if pred_ret > 0 else "red"
        fig.add_trace(go.Scatter(
            x=[df.index[-1] + timedelta(hours=1)],
            y=[pred_price],
            mode="markers+text",
            marker=dict(size=14, color=color),
            name="Predicted Next Hour",
            text=["Prediction"],
            textposition="top center"
        ))
    fig.update_layout(
        xaxis_title="Datetime",
        yaxis_title="Price",
        showlegend=True,
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    # -- Prediction Result Card --
    if predict_btn and last_feat_row is not None:
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        # Predicted Price
        col1.metric("Predicted Price (Next Hour)", f"${pred_price:.2f}")
        # Direction & Magnitude
        pct_change = (pred_price - current_price) / current_price * 100
        direction = "UP" if pred_price > current_price else ("DOWN" if pred_price < current_price else "HOLD")
        col2.metric("Predicted Direction", f"{direction} ({pct_change:+.2f}%)")
        # Confidence
        col3.progress(confidence, text=f"Model Confidence: {int(confidence*100)}%")
        # Ensemble Votes
        vote_text = f"CNN: {votes['CNN']} | LSTM: {votes['LSTM']} | XGBoost: {votes['XGBoost']}"
        col4.info(f"**Ensemble Votes**\n{vote_text}")

        # -- Trading Suggestion --
        st.subheader("🚦 Trading Signal & Reasoning")
        # Simple strat: if abs(pred_ret) > 0.3%, suggest BUY/SELL, else HOLD
        threshold = 0.3
        if abs(pct_change) < threshold:
            signal = "HOLD"
            reason = f"Predicted change ({pct_change:+.2f}%) is below actionable threshold ({threshold}%)."
        elif direction == "UP":
            signal = "STRONG BUY" if confidence > 0.75 else "BUY"
            reason = f"Model predicts a {pct_change:+.2f}% increase next hour, confidence {int(confidence*100)}%."
        else:
            signal = "STRONG SELL" if confidence > 0.75 else "SELL"
            reason = f"Model predicts a {pct_change:+.2f}% decrease next hour, confidence {int(confidence*100)}%."
        st.success(f"**SIGNAL:** {signal}")
        st.caption(f"**Reasoning:** {reason}")

        # -- LOG THE PREDICTION (Append to session_state) --
        st.session_state["prediction_log"].append({
            "Timestamp": df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
            "Ticker": ticker,
            "Price at Prediction": current_price,
            "Predicted Price (1H)": pred_price,
            "Direction (Predicted)": direction,
            "Model Confidence": int(confidence*100),
            "Ensemble Votes": vote_text,
            "CNN": pred_cnn, "LSTM": pred_lstm, "XGB": pred_xgb,
            "Meta": pred_ret
        })

# -- SECTION 2: SIMULATION & PERFORMANCE TRACKING --

st.header("📊 Trading Simulation Log & Performance")

# Prediction Log Table
log_df = pd.DataFrame(st.session_state["prediction_log"])
if not log_df.empty:
    # If actual price available for next hour, fetch it and compute metrics
    actual_prices = []
    directions_actual = []
    was_corrects = []
    profits = []

    for i, row in log_df.iterrows():
        try:
            history = get_stock_data(row["Ticker"], 2)
            idx_now = pd.to_datetime(row["Timestamp"])
            idx_next = idx_now + timedelta(hours=1)
            actual_price = history.loc[idx_next:, "Close"].iloc[0] if idx_next in history.index else np.nan
            actual_prices.append(actual_price)
            if np.isnan(actual_price):
                directions_actual.append("N/A")
                was_corrects.append("N/A")
                profits.append(np.nan)
            else:
                dir_actual = "UP" if actual_price > row["Price at Prediction"] else ("DOWN" if actual_price < row["Price at Prediction"] else "HOLD")
                directions_actual.append(dir_actual)
                was_corrects.append("Yes" if dir_actual == row["Direction (Predicted)"] else "No")
                profit = (actual_price - row["Price at Prediction"]) / row["Price at Prediction"] * 100 if dir_actual == "UP" else \
                    (row["Price at Prediction"] - actual_price) / row["Price at Prediction"] * 100 if dir_actual == "DOWN" else 0
                profits.append(profit)
        except Exception:
            actual_prices.append(np.nan)
            directions_actual.append("N/A")
            was_corrects.append("N/A")
            profits.append(np.nan)

    log_df["Actual Price (1H Later)"] = actual_prices
    log_df["Direction (Actual)"] = directions_actual
    log_df["Was Correct?"] = was_corrects
    log_df["Profit (%)"] = profits

    st.dataframe(log_df, use_container_width=True)

    # Performance Metrics
    st.subheader("Performance Dashboard")
    colA, colB, colC, colD = st.columns(4)
    # Total Return (%)
    total_return = np.nansum(log_df["Profit (%)"].values)
    colA.metric("Total Return (%)", f"{total_return:.2f}%")
    # Accuracy (%)
    num_correct = np.sum(log_df["Was Correct?"] == "Yes")
    accuracy = num_correct / len(log_df) * 100 if len(log_df) > 0 else 0
    colB.metric("Accuracy (%)", f"{accuracy:.2f}%")
    # Win Rate (%)
    num_profitable = np.sum(log_df["Profit (%)"] > 0)
    win_rate = num_profitable / len(log_df) * 100 if len(log_df) > 0 else 0
    colC.metric("Win Rate (%)", f"{win_rate:.2f}%")
    # Avg Profit/trade
    avg_profit = np.nanmean(log_df["Profit (%)"].values) if len(log_df) > 0 else 0
    colD.metric("Avg Profit per Trade (%)", f"{avg_profit:.3f}%")

    # Equity Curve
    st.subheader("Virtual Portfolio Growth")
    base = 10000
    equity_curve = np.nancumsum(np.nan_to_num(log_df["Profit (%)"].values) * base / 100) + base
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=log_df["Timestamp"], y=equity_curve,
        mode="lines+markers", name="Equity"
    ))
    fig_eq.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="Portfolio Value ($)",
        height=350
    )
    st.plotly_chart(fig_eq, use_container_width=True)
