import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import datetime
import plotly.graph_objects as go
import os

# --- Streamlit page config ---
st.set_page_config(
    page_title="SOLARIS : A Machine Learning Based Hourly Stock Prediction Tool",
    page_icon=":sun_with_face:",
    layout="wide",
)

# --- Constants ---
START_CAPITAL = 10000

# --- Load Models & Scaler ---
@st.cache_data(show_spinner=False)
def load_models():
    cnn_model = joblib.load("cnn_model.pkl")
    lstm_model = joblib.load("lstm_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    meta_model = joblib.load("meta_learner.pkl")
    scaler = joblib.load("scaler.pkl")
    return cnn_model, lstm_model, xgb_model, meta_model, scaler

try:
    cnn_model, lstm_model, xgb_model, meta_model, scaler = load_models()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model loading error: {e}")

# --- Data Gathering & Preprocessing (based on typical notebook workflow) ---
@st.cache_data(show_spinner=True)
def get_yf_data(ticker, ndays=7):
    interval = "60m"
    period = f"{ndays}d"
    df = yf.download(ticker, interval=interval, period=period)
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    return df

def make_features(df):
    feat = pd.DataFrame(index=df.index)
    price_series = df['Close']

    # Lag returns
    for lag in [1, 3, 6, 12, 24]:
        feat[f"ret_{lag}h"] = price_series.pct_change(lag)
    # Rolling volatility
    for window in [6, 12, 24]:
        feat[f"vol_{window}h"] = price_series.pct_change().rolling(window).std()
    # RSI
    delta = price_series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    feat["rsi_14"] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = price_series.ewm(span=12, adjust=False).mean()
    ema26 = price_series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    feat["macd"] = macd
    feat["macd_signal"] = signal
    # Moving averages
    for w in [5, 10, 20]:
        feat[f"sma_{w}"] = price_series.rolling(w).mean()
        feat[f"ema_{w}"] = price_series.ewm(span=w, adjust=False).mean()
    # Volume
    if "Volume" in df.columns:
        vol_series = df["Volume"]
        feat["vol_change_1h"] = vol_series.pct_change()
        feat["vol_ma_24h"] = vol_series.rolling(24).mean()
    # Calendar features
    feat["hour"] = feat.index.hour
    feat["day_of_week"] = feat.index.dayofweek
    # Drop NA and keep last row for prediction
    feat = feat.dropna().tail(1)
    return feat

# --- Utility: Prediction Pipeline ---
def predict_next_hour(ticker):
    df = get_yf_data(ticker, ndays=6)
    if len(df) < 128:
        return None, "Not enough hourly data for prediction.", None
    features = make_features(df)
    if features.shape[0] == 0:
        return None, "Not enough valid features for prediction.", None
    X_tab = features.values
    X_tab_scaled = scaler.transform(X_tab)
    # Sequence models: use last 128 rows for LSTM/CNN
    seq_features = []
    for i in range(128):
        f = make_features(df.iloc[:-(128-i)])
        if f.shape[0] == 0:
            continue
        seq_features.append(f.values[0])
    if len(seq_features) < 128:
        seq_features = [X_tab_scaled[0]] * 128
    X_seq = np.array(seq_features).reshape(1, 128, X_tab_scaled.shape[1])
    # Predict
    pred_cnn = cnn_model.predict(X_seq, verbose=0)[0][0]
    pred_lstm = lstm_model.predict(X_seq, verbose=0)[0][0]
    pred_xgb = xgb_model.predict(X_tab_scaled)[0]
    meta_X = np.array([[pred_cnn, pred_lstm, pred_xgb,
                        np.mean([pred_cnn, pred_lstm, pred_xgb]),
                        np.std([pred_cnn, pred_lstm, pred_xgb]),
                        np.max([pred_cnn, pred_lstm, pred_xgb]),
                        np.min([pred_cnn, pred_lstm, pred_xgb])]])
    pred_meta = meta_model.predict(meta_X)[0]
    current_price = df.iloc[-1]['Close']
    predicted_price = current_price * (1 + pred_meta)
    pct_change = pred_meta * 100
    direction = "UP" if pred_meta > 0 else "DOWN" if pred_meta < 0 else "HOLD"
    confidence = max(0, min(100, 100 - np.std([pred_cnn, pred_lstm, pred_xgb]) * 9000))
    votes = {
        "CNN": "UP" if pred_cnn > 0 else "DOWN" if pred_cnn < 0 else "HOLD",
        "LSTM": "UP" if pred_lstm > 0 else "DOWN" if pred_lstm < 0 else "HOLD",
        "XGBoost": "UP" if pred_xgb > 0 else "DOWN" if pred_xgb < 0 else "HOLD"
    }
    return {
        "predicted_price": predicted_price,
        "current_price": current_price,
        "pred_meta": pred_meta,
        "pct_change": pct_change,
        "direction": direction,
        "confidence": confidence,
        "votes": votes,
        "pred_cnn": pred_cnn,
        "pred_lstm": pred_lstm,
        "pred_xgb": pred_xgb
    }, None, df

# --- Virtual Trading Log ---
if "trade_log" not in st.session_state:
    st.session_state.trade_log = []

def add_trade_log_entry(entry):
    st.session_state.trade_log.append(entry)

def get_trade_log_df():
    return pd.DataFrame(st.session_state.trade_log)

# --- App Layout ---
st.title("SOLARIS : A Machine Learning Based Hourly Stock Prediction Tool")
st.subheader("A ML Model combining CNN, LSTM, and XGBoost with a Meta Learner Ensemble for Short Term Stock Market Prediction")

st.markdown("""
This project explores the potential of ensemble deep learning models to identify short-term patterns in financial time series data for hourly price prediction.  
**Note:** This tool is for educational and research purposes only. Not financial advice!
""")

st.header("1. Data Gathering & Preprocessing")

st.markdown("""
**Data Source:** Yahoo Finance  
**Preprocessing Steps:**  
- Download hourly OHLCV data for selected ticker  
- Feature engineering: lag returns, volatility, RSI, MACD, moving averages, volume features  
- Calendar features: hour, day of week  
""")

with st.expander("Show raw data & engineered features"):
    ticker_demo = st.text_input("Preview ticker (e.g., AAPL, TSLA, MSFT):", value="AAPL")
    df_raw = get_yf_data(ticker_demo, ndays=7)
    st.dataframe(df_raw.tail(10))
    st.write("Engineered Features (last row):")
    st.dataframe(make_features(df_raw))

st.divider()

# === SECTION 2: LIVE PREDICTION DASHBOARD ===
st.header("2. Live Prediction Dashboard")

col1, col2 = st.columns([2, 1])
with col1:
    ticker = st.text_input("Enter Stock Ticker for Prediction:", value="AAPL", key="pred_ticker")
    ndays = st.selectbox("Number of historical days to display", options=[1, 7, 30], index=1)
    predict_btn = st.button("🚀 Predict Next Hour")

if predict_btn and ticker and model_loaded:
    with st.spinner(f"Predicting next hour for {ticker}..."):
        result, error, df = predict_next_hour(ticker)
    if error:
        st.error(error)
    else:
        st.subheader(f"Live Price Chart for {ticker}")
        df_plot = get_yf_data(ticker, ndays)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_plot.index,
            open=df_plot['Open'],
            high=df_plot['High'],
            low=df_plot['Low'],
            close=df_plot['Close'],
            name='Price'
        ))
        now = df_plot.index[-1]
        fig.add_vline(x=now, line_dash="dash", line_color="blue", annotation_text="Now", annotation_position="top left")
        pred_time = now + pd.Timedelta(hours=1)
        fig.add_trace(go.Scatter(
            x=[pred_time],
            y=[result["predicted_price"]],
            mode="markers+text",
            marker=dict(color="green" if result["direction"] == "UP" else "red", size=12),
            text=[f'Pred: ${result["predicted_price"]:.2f}'],
            textposition="bottom center",
            name='Prediction'
        ))
        fig.update_layout(height=400, xaxis_title="Time", yaxis_title="Price ($)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Prediction Result")
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        pcol1.metric("Predicted Price (1H)", f"${result['predicted_price']:.2f}")
        pcol2.metric("Direction", f"{result['direction']} ({result['pct_change']:+.2f}%)")
        pcol3.progress(int(result['confidence']), text=f"Model Confidence: {int(result['confidence'])}%")
        pcol4.markdown(f"**Ensemble Votes:**<br>"
            f"<span style='color:green'>CNN: {result['votes']['CNN']}</span><br>"
            f"<span style='color:blue'>LSTM: {result['votes']['LSTM']}</span><br>"
            f"<span style='color:purple'>XGBoost: {result['votes']['XGBoost']}</span>", unsafe_allow_html=True)

        threshold = 0.2
        if abs(result['pct_change']) > threshold:
            signal = "STRONG BUY" if result['direction'] == "UP" else "STRONG SELL"
        elif abs(result['pct_change']) > 0.05:
            signal = "BUY" if result['direction'] == "UP" else "SELL"
        else:
            signal = "HOLD"
        st.info(f"**SIGNAL:** {signal}\n\n"
                f"The model predicts a {result['pct_change']:+.2f}% {'increase' if result['direction']=='UP' else 'decrease'} in the next hour.")

        entry = {
            "Timestamp": now,
            "Ticker": ticker,
            "Price at Prediction": result["current_price"],
            "Predicted Price (1H)": result["predicted_price"],
            "Direction (Predicted)": result["direction"],
            "Model Confidence (%)": int(result["confidence"]),
            "CNN": result["votes"]["CNN"],
            "LSTM": result["votes"]["LSTM"],
            "XGBoost": result["votes"]["XGBoost"],
            "Signal": signal,
            "Profit (%)": None,
            "Actual Price (1H Later)": None,
            "Direction (Actual)": None,
            "Was Correct?": None
        }
        add_trade_log_entry(entry)

st.divider()

# === SECTION 3: SIMULATION & PERFORMANCE TRACKING ===
st.header("3. Trading Simulation Log & Performance")

trade_df = get_trade_log_df()
if not trade_df.empty:
    for idx, row in trade_df.iterrows():
        if pd.isnull(row.get("Actual Price (1H Later)")):
            ts = row["Timestamp"]
            ticker = row["Ticker"]
            df_actual = get_yf_data(ticker, ndays=2)
            actual_row = df_actual[df_actual.index == ts + pd.Timedelta(hours=1)]
            if not actual_row.empty:
                actual_price = actual_row.iloc[0]['Close']
                pred_price = row["Predicted Price (1H)"]
                direction_actual = "UP" if actual_price > row["Price at Prediction"] else "DOWN" if actual_price < row["Price at Prediction"] else "HOLD"
                was_correct = (row["Direction (Predicted)"] == direction_actual)
                profit = ((actual_price - row["Price at Prediction"]) / row["Price at Prediction"]) * 100 if row["Signal"] != "HOLD" else 0
                st.session_state.trade_log[idx]["Actual Price (1H Later)"] = actual_price
                st.session_state.trade_log[idx]["Direction (Actual)"] = direction_actual
                st.session_state.trade_log[idx]["Was Correct?"] = "Yes" if was_correct else "No"
                st.session_state.trade_log[idx]["Profit (%)"] = profit
    trade_df = get_trade_log_df()
    st.dataframe(trade_df, use_container_width=True, height=300)
    total_return = trade_df["Profit (%)"].dropna().sum()
    accuracy = trade_df["Was Correct?"].dropna().eq("Yes").mean() * 100 if "Was Correct?" in trade_df else 0
    num_trades = trade_df["Signal"].dropna().apply(lambda x: x != "HOLD").sum()
    win_rate = trade_df["Profit (%)"].dropna().apply(lambda x: x > 0).mean() * 100 if num_trades > 0 else 0
    avg_profit = trade_df["Profit (%)"].dropna().mean()
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Total Return (%)", f"{total_return:.2f}%")
    mcol2.metric("Accuracy (%)", f"{accuracy:.2f}%")
    mcol3.metric("Win Rate (%)", f"{win_rate:.2f}%")
    mcol4.metric("Avg Profit per Trade (%)", f"{avg_profit:.2f}%")
    equity_curve = [START_CAPITAL]
    for _, row in trade_df.iterrows():
        profit = row.get("Profit (%)", 0)
        equity_curve.append(equity_curve[-1] * (1 + profit/100) if profit is not None else equity_curve[-1])
    equity_curve = equity_curve[1:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=equity_curve, mode="lines", name="Portfolio Value"))
    fig.update_layout(xaxis_title="Trade #", yaxis_title="Portfolio Value ($)", height=300)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# === SECTION 4: MODEL EXPLANATION & TECHNICAL DETAILS ===
st.header("4. Model Explanation & Technical Details")

exp_col1, exp_col2 = st.columns([2,1])
with exp_col1:
    st.subheader("How the Meta-Learner Works")
    st.markdown("""
    **Flow:**  
    Input Data → CNN Block → LSTM Block → XGBoost → Meta-Learner (Stacked/Averaged) → Final Prediction  
    - **CNN:** Pattern recognition in short-term price movement  
    - **LSTM:** Captures sequential dependencies and "memory" of time series  
    - **XGBoost:** Handles tabular/structured features  
    - **Meta-Learner:** Weighted/stacked ensemble for final output
    """)
with exp_col2:
    st.subheader("Feature Importance (XGBoost)")
    try:
        feat_names = [col for col in make_features(get_yf_data("AAPL", 2)).columns]
        importances = xgb_model.feature_importances_
        top_idx = np.argsort(importances)[-10:]
        top_feats = [feat_names[i] for i in top_idx]
        top_vals = importances[top_idx]
        imp_fig = go.Figure(go.Bar(x=top_vals, y=top_feats, orientation="h"))
        imp_fig.update_layout(height=300, xaxis_title="Importance")
        st.plotly_chart(imp_fig, use_container_width=True)
    except Exception as e:
        st.info("Feature importances not available for XGBoost model.")

st.subheader("Technical Stack")
st.markdown("""
- **Streamlit:** App UI & dashboard
- **yfinance:** Hourly price data
- **CNN, LSTM, XGBoost:** Model ensemble
- **Meta Learner:** Final prediction
- **Plotly:** Interactive charts
""")
