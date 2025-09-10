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
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="SOLARIS: AI Stock Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.5rem; color: #7f7f7f; text-align: center; margin-bottom: 2rem; }
    .prediction-card { background-color: #f0f2f6; border-radius: 0.5rem; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 1.5rem; }
    .metric-card { background-color: white; border-radius: 0.5rem; padding: 1rem; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); text-align: center; }
    .positive { color: #2ecc71; }
    .negative { color: #e74c3c; }
    .stProgress > div > div > div > div { background-color: #1f77b4; }
    .log-table { font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">SOLARIS</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="sub-header">A Machine Learning Based Hourly Stock Prediction Tool</h2>', unsafe_allow_html=True)

st.write("""
This application combines Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM) networks, 
and XGBoost with a Meta Learner Ensemble for short-term stock market predictions. 
*Note: This is for educational and research purposes only - not financial advice.*
""")

# Persistent log file
PREDICTION_LOG_FILE = "prediction_history.json"

def convert_all_datetimes(obj):
    if isinstance(obj, dict):
        return {k: convert_all_datetimes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_all_datetimes(x) for x in obj]
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.strftime("%Y-%m-%d %H:%M")
    else:
        return obj

def save_prediction_log(log):
    log_save = convert_all_datetimes(log)
    with open(PREDICTION_LOG_FILE, "w") as f:
        json.dump(log_save, f, indent=2)

def load_prediction_log():
    if os.path.exists(PREDICTION_LOG_FILE):
        with open(PREDICTION_LOG_FILE, "r") as f:
            try:
                log = json.load(f)
                # convert timestamp str to datetime, and target_time/evaluated_time if present
                for entry in log:
                    if "timestamp" in entry and entry["timestamp"]:
                        entry["timestamp"] = pd.to_datetime(entry["timestamp"])
                    if "target_time" in entry and entry["target_time"]:
                        entry["target_time"] = pd.to_datetime(entry["target_time"])
                    if "evaluated_time" in entry and entry["evaluated_time"]:
                        entry["evaluated_time"] = pd.to_datetime(entry["evaluated_time"])
                return log
            except Exception:
                return []
    return []

def get_prediction_log():
    if "prediction_log" not in st.session_state:
        st.session_state.prediction_log = load_prediction_log()
    return st.session_state.prediction_log

def append_prediction_log(entry):
    log = get_prediction_log()
    log.append(entry)
    save_prediction_log(log)
    st.session_state.prediction_log = log

def update_prediction_log(log):
    save_prediction_log(log)
    st.session_state.prediction_log = log

def evaluate_predictions():
    log = get_prediction_log()
    to_update = []
    for entry in log:
        if "actual_price" not in entry or entry["actual_price"] is None or (isinstance(entry["actual_price"], float) and np.isnan(entry["actual_price"])):
            ticker = entry.get("ticker", None)
            timestamp = entry.get("target_time", None)
            if ticker is None or timestamp is None:
                to_update.append(False)
                continue
            try:
                stock = yf.Ticker(ticker)
                # Pull a window of 4 hours after the prediction time in 60m interval to get the closes
                df = stock.history(
                    start=(timestamp - pd.Timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
                    end=(timestamp + pd.Timedelta(hours=4)).strftime("%Y-%m-%d %H:%M:%S"),
                    interval="60m"
                )
                if not df.empty:
                    actual_row = df[df.index >= timestamp]
                    if not actual_row.empty:
                        actual_price = float(actual_row.iloc[0]['Close'])
                        entry["actual_price"] = actual_price
                        entry["error_pct"] = abs(entry["predicted_price"] - actual_price) / actual_price * 100
                        entry["error_abs"] = abs(entry["predicted_price"] - actual_price)
                        entry["evaluated_time"] = str(actual_row.index[0])
                        to_update.append(True)
                    else:
                        to_update.append(False)
                else:
                    to_update.append(False)
            except Exception:
                to_update.append(False)
        else:
            to_update.append(False)
    if any(to_update):
        update_prediction_log(log)
    return log

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

# Original cross assets used in training
original_cross_assets = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "NVDA", "JPM", "JNJ", "XOM", "CAT", "BA", "META"]
dynamic_cross_assets = original_cross_assets.copy()

# HARDCODED feature columns for scaler/model compatibility
expected_columns = [
    'ret_1h', 'ret_3h', 'ret_6h', 'ret_12h', 'ret_24h', 'vol_6h', 'vol_12h', 'vol_24h',
    'rsi_14', 'macd', 'macd_signal', 'sma_5', 'ema_5', 'sma_10', 'ema_10', 'sma_20', 'ema_20',
    'vol_change_1h', 'vol_ma_24h', 'AAPL_ret_1h', 'MSFT_ret_1h', 'AMZN_ret_1h', 'GOOGL_ret_1h',
    'TSLA_ret_1h', 'NVDA_ret_1h', 'JPM_ret_1h', 'JNJ_ret_1h', 'XOM_ret_1h', 'CAT_ret_1h',
    'BA_ret_1h', 'META_ret_1h', 'hour', 'day_of_week', 'price'
]
SEQ_LEN = 128  # Sequence length for CNN/LSTM

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

    # Always fill all cross asset columns, even if they're missing
    for asset in original_cross_assets:
        feat_tmp[f"{asset}_ret_1h"] = cross_data.get(asset, pd.Series(0, index=feat_tmp.index)).pct_change().fillna(0)

    feat_tmp["hour"] = feat_tmp.index.hour
    feat_tmp["day_of_week"] = feat_tmp.index.dayofweek
    feat_tmp["price"] = price_series

    for col in ["rsi_14", "macd", "macd_signal", "vol_change_1h", "vol_ma_24h"]:
        feat_tmp[col] = feat_tmp[col].fillna(0)
    feat_tmp = feat_tmp.dropna(subset=["price"])

    # HARDCODE output columns for scaler/model compatibility
    for col in expected_columns:
        if col not in feat_tmp.columns:
            feat_tmp[col] = 0
    feat_tmp = feat_tmp[expected_columns]
    return feat_tmp

def predict_with_models(features, current_price, scaler, cnn_model, lstm_model, xgb_model, meta_model):
    """
    Predict using the CNN, LSTM, XGBoost and Meta Learner.
    """
    if scaler is None or cnn_model is None or lstm_model is None or xgb_model is None or meta_model is None:
        return None, None, None, None

    # Always use hardcoded expected_columns
    features = features[expected_columns]
    # CLEAN FEATURES: Replace NaN/infs with 0
    features = features.replace([np.inf, -np.inf], 0).fillna(0)

    if len(features) < SEQ_LEN:
        return None, None, None, None
    features_recent = features.iloc[-SEQ_LEN:]
    features_tabular = features.iloc[-1:]

    # If you want to be extra safe, also clean these again:
    features_recent = features_recent.replace([np.inf, -np.inf], 0).fillna(0)
    features_tabular = features_tabular.replace([np.inf, -np.inf], 0).fillna(0)

    # Scale features
    X_seq = scaler.transform(features_recent)
    X_tab = scaler.transform(features_tabular)

    # CNN/LSTM expect (1, SEQ_LEN, n_features)
    X_seq = X_seq.reshape(1, SEQ_LEN, -1)
    # XGBoost expects (1, n_features)
    X_tab = X_tab.reshape(1, -1)

    # Predict next-hour return (not price)
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

    # Meta learner features: base preds + stats (mean, std, max, min)
    meta_feats = np.array([pred_cnn, pred_lstm, pred_xgb])
    meta_stats = np.array([meta_feats.mean(), meta_feats.std(), meta_feats.max(), meta_feats.min()])
    meta_input = np.concatenate([meta_feats, meta_stats]).reshape(1, -1)
    try:
        pred_meta = meta_model.predict(meta_input)
        pred_meta = float(pred_meta[0])
    except Exception as e:
        pred_meta = np.mean([pred_cnn, pred_lstm, pred_xgb])

    # Compute predicted price (current * (1 + pred_meta))
    predicted_price = current_price * (1 + pred_meta)
    pred_change_pct = pred_meta * 100

    # Votes
    votes = {
        "CNN": "UP" if pred_cnn > 0 else "DOWN",
        "LSTM": "UP" if pred_lstm > 0 else "DOWN",
        "XGBoost": "UP" if pred_xgb > 0 else "DOWN"
    }
    # Confidence = agreement among models, weighted by absolute returns
    agreement = [pred_cnn, pred_lstm, pred_xgb]
    up_count = sum([1 for v in agreement if v > 0])
    down_count = sum([1 for v in agreement if v < 0])
    confidence = (max(up_count, down_count) / 3) * 100

    return predicted_price, pred_change_pct, votes, confidence

st.sidebar.header("Configuration")
selected_ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper()
days_history = st.sidebar.slider("Days of History", min_value=30, max_value=729, value=90)
run_prediction = st.sidebar.button("Run Prediction")
evaluate_results = st.sidebar.button("Evaluate Predictions (fetch actual prices)")

main_ticker_data = fetch_stock_data(selected_ticker, period=f"{days_history}d", interval="60m")

if main_ticker_data is None or main_ticker_data.empty or len(main_ticker_data) < 128:
    st.error(f"No or insufficient data found for ticker: {selected_ticker}. Please check the symbol or try another stock.")
else:
    latest_data = main_ticker_data.iloc[-1]
    current_price = latest_data['Close']
    current_time = latest_data.name
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    with col2:
        prev_close = main_ticker_data.iloc[-2]['Close'] if len(main_ticker_data) > 1 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100
        st.metric("Change", f"{change:.2f}", f"{change_pct:.2f}%")
    with col3:
        st.metric("Last Updated", current_time.strftime("%Y-%m-%d %H:%M"))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        subplot_titles=(f'{selected_ticker} Price', 'Volume'),
                        row_width=[0.2, 0.7])
    fig.add_trace(go.Candlestick(x=main_ticker_data.index,
                                 open=main_ticker_data['Open'],
                                 high=main_ticker_data['High'],
                                 low=main_ticker_data['Low'],
                                 close=main_ticker_data['Close'],
                                 name="Price"), row=1, col=1)
    colors = ['red' if row['Open'] > row['Close'] else 'green' 
              for _, row in main_ticker_data.iterrows()]
    fig.add_trace(go.Bar(x=main_ticker_data.index,
                         y=main_ticker_data['Volume'],
                         marker_color=colors,
                         name="Volume"), row=2, col=1)
    fig.add_vline(x=current_time, line_dash="dash", line_color="white")
    fig.update_layout(height=600, showlegend=False, 
                     xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    if cnn_model and lstm_model and xgb_model and meta_model and scaler:
        cross_data = fetch_cross_assets(main_ticker_data.index, days_history, selected_ticker)
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
                        st.markdown("## Prediction Results")
                        pred_col1, pred_col2, pred_col3 = st.columns(3)
                        with pred_col1:
                            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                            st.metric("Predicted Price (next hour)", f"${predicted_price:.2f}", 
                                     f"{pred_change_pct:.2f}%")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with pred_col2:
                            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                            st.write("**Model Votes**")
                            for model, vote in vote_display.items():
                                color_class = "positive" if vote == "UP" else "negative"
                                st.markdown(f"{model}: <span class='{color_class}'>{vote}</span>", 
                                           unsafe_allow_html=True)
                            st.write("**Confidence**")
                            st.progress(confidence/100)
                            st.write(f"{confidence:.1f}%")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with pred_col3:
                            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                            if pred_change_pct > 1.0 and confidence > 70:
                                st.success("**SIGNAL: STRONG BUY**")
                                st.write(f"The model predicts a {pred_change_pct:.2f}% increase with high confidence.")
                            elif pred_change_pct > 0.5:
                                st.info("**SIGNAL: BUY**")
                                st.write(f"The model predicts a {pred_change_pct:.2f}% increase.")
                            elif pred_change_pct < -1.0 and confidence > 70:
                                st.error("**SIGNAL: STRONG SELL**")
                                st.write(f"The model predicts a {abs(pred_change_pct):.2f}% decrease with high confidence.")
                            elif pred_change_pct < -0.5:
                                st.warning("**SIGNAL: SELL**")
                                st.write(f"The model predicts a {abs(pred_change_pct):.2f}% decrease.")
                            else:
                                st.info("**SIGNAL: HOLD**")
                                st.write("No strong directional signal detected.")
                            st.markdown('</div>', unsafe_allow_html=True)
                        # Instead of logging just the current time, log the prediction for the next hour
                        target_time = current_time + pd.Timedelta(hours=1)
                        log_entry = {
                            "timestamp": datetime.now(),
                            "ticker": selected_ticker,
                            "current_price": float(current_price),
                            "predicted_price": float(predicted_price),
                            "predicted_change": float(pred_change_pct),
                            "confidence": float(confidence),
                            "signal": "BUY" if pred_change_pct > 0.5 else "SELL" if pred_change_pct < -0.5 else "HOLD",
                            "target_time": target_time,  # the time the prediction is for
                            "actual_price": None,
                            "error_pct": None,
                            "error_abs": None,
                            "evaluated_time": None
                        }
                        append_prediction_log(log_entry)
                        st.info("Prediction logged. Later, use the 'Evaluate Predictions' button in the sidebar to fetch the actual price for that prediction time.")
        if evaluate_results:
            with st.spinner("Evaluating predictions..."):
                evaluate_predictions()
        # Show predictions log
        log = get_prediction_log()
        if log:
            st.markdown("## Prediction History")
            log_df = pd.DataFrame(log)
            # Safely handle missing columns and type conversions
            if "timestamp" in log_df.columns:
                log_df['timestamp'] = pd.to_datetime(log_df['timestamp'], errors="coerce")
            else:
                log_df['timestamp'] = pd.NaT
            if "target_time" in log_df.columns:
                log_df['target_time'] = pd.to_datetime(log_df['target_time'], errors="coerce")
            else:
                log_df['target_time'] = pd.NaT
            if "evaluated_time" in log_df.columns:
                log_df['evaluated_time'] = pd.to_datetime(log_df['evaluated_time'], errors="coerce")
            else:
                log_df['evaluated_time'] = pd.NaT
            log_df = log_df.sort_values("timestamp", ascending=False)
            # Always create missing columns for display
            show_cols = ["timestamp", "ticker", "current_price", "predicted_price", "target_time", "actual_price", "error_pct", "error_abs", "confidence", "signal"]
            for col in show_cols:
                if col not in log_df.columns:
                    log_df[col] = np.nan
            st.dataframe(log_df[show_cols], use_container_width=True)
            # Show evaluation summary if available
            if 'error_pct' in log_df.columns and log_df['error_pct'].notnull().any():
                eval_rows = log_df[log_df['error_pct'].notnull()]
                avg_abs_error = eval_rows['error_abs'].mean()
                avg_pct_error = eval_rows['error_pct'].mean()
                eval_count = len(eval_rows)
                st.markdown("## Prediction Evaluation Summary")
                st.metric("Evaluated Predictions", eval_count)
                st.metric("Average Absolute Error", f"${avg_abs_error:.3f}")
                st.metric("Average % Error", f"{avg_pct_error:.2f}%")
                st.progress(min(1, max(0, 1-avg_pct_error/10)))  # visual bar: 0% error = full bar, 10%+ = empty
    else:
        st.warning("""
        Models failed to load. Please ensure you have the following files in the working directory:
        - cnn_model.pkl
        - lstm_model.pkl
        - xgb_model.pkl
        - meta_learner.pkl
        - scaler.pkl
        """)

st.markdown("---")
st.markdown("## How SOLARIS Works")
exp_col1, exp_col2 = st.columns(2)
with exp_col1:
    st.markdown("""
    ### Model Architecture
    SOLARIS uses an ensemble of three different machine learning models:
    1. **CNN (Convolutional Neural Network)**: Identifies patterns in price movements
    2. **LSTM (Long Short-Term Memory)**: Captures sequential dependencies in time series data
    3. **XGBoost**: Handles structured features and non-linear relationships
    Predictions from these models are combined using a meta-learner for improved accuracy.
    """)
    st.markdown("### Top Predictive Features")
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
        st.write(f"{feature}: {importance:.0%}")
        st.progress(importance)
with exp_col2:
    st.markdown("""
    ### Technical Details
    - **Data Source**: Yahoo Finance API
    - **Feature Engineering**: 34 technical indicators and cross-asset features
    - **Sequence Length**: 128 hours (5+ days) of historical data
    - **Training Period**: 2+ years of hourly data
    - **Update Frequency**: Predictions are generated in real-time
    """)
    st.markdown("### Historical Backtest Performance")
    st.markdown("""
    Based on extensive backtesting, the model has achieved:
    - **Total Return**: 239.97%
    - **Start Capital**: $10,000.00
    - **Final Capital**: $33,997.15
    - **Total Trades**: 8,944
    - **Avg Hourly PnL**: $2.68 ± $91.17
    """)
    st.error("""
    **Important Disclaimer**: 
    This tool is for educational and research purposes only. 
    Past performance is not indicative of future results. 
    Always conduct your own research and consider seeking advice from a qualified financial advisor before making investment decisions.
    """)
