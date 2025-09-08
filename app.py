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

if 'prediction_log' not in st.session_state:
    st.session_state.prediction_log = []
if 'performance_metrics' not in st.session_state:
    st.session_state.performance_metrics = {
        'total_predictions': 0,
        'correct_predictions': 0,
        'total_return': 0,
        'accuracy': 0
    }

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

# HARDCODED feature column order based on your notebook's X.columns.tolist()
feature_columns = [
    "price",
    "ret_1h", "ret_3h", "ret_6h", "ret_12h", "ret_24h",
    "vol_6h", "vol_12h", "vol_24h",
    "rsi_14", "macd", "macd_signal",
    "sma_5", "sma_10", "sma_20",
    "ema_5", "ema_10", "ema_20",
    "vol_change_1h", "vol_ma_24h",
    "AAPL_ret_1h", "MSFT_ret_1h", "AMZN_ret_1h", "GOOGL_ret_1h", "TSLA_ret_1h",
    "NVDA_ret_1h", "JPM_ret_1h", "JNJ_ret_1h", "XOM_ret_1h", "CAT_ret_1h", "BA_ret_1h", "META_ret_1h",
    "hour", "day_of_week"
]

st.sidebar.header("Configuration")
selected_ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL").upper()
days_history = st.sidebar.text_input("Days of Historical Data (90-729)", value="180")

try:
    days_history = int(days_history)
    if days_history < 90 or days_history > 729:
        st.sidebar.error("Please enter a value between 90 and 729")
        days_history = 180
except ValueError:
    st.sidebar.error("Please enter a valid number")
    days_history = 180

run_prediction = st.sidebar.button("🚀 Predict Next Hour")
update_predictions = st.sidebar.button("🔄 Update Prediction Results")

@st.cache_data(ttl=300)
def fetch_stock_data(ticker, period="30d", interval="60m"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            st.error(f"No data found for ticker {ticker}")
            return None
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def get_aligned_ffill(tickers, days_history):
    all_data = {}
    for t in tickers:
        df = fetch_stock_data(t, period=f"{days_history}d", interval="60m")
        if df is not None and not df.empty:
            all_data[t] = df
    if selected_ticker not in all_data:
        return None, None
    target_index = all_data[selected_ticker].index
    aligned_close = pd.DataFrame(index=target_index)
    for t, df in all_data.items():
        aligned_close[t] = df.reindex(target_index)['Close']
    aligned_ffill = aligned_close.ffill()
    return aligned_ffill, all_data

def create_features_for_app(selected_ticker, aligned_ffill, data_dict):
    price_series = aligned_ffill[selected_ticker]
    feat_tmp = pd.DataFrame(index=price_series.index)

    # Price column
    feat_tmp["price"] = price_series

    # Lag returns
    for lag in [1, 3, 6, 12, 24]:
        feat_tmp[f"ret_{lag}h"] = price_series.pct_change(lag)

    # Rolling volatility
    for window in [6, 12, 24]:
        feat_tmp[f"vol_{window}h"] = price_series.pct_change().rolling(window).std()

    # Technical indicators
    try:
        feat_tmp["rsi_14"] = ta.momentum.RSIIndicator(price_series, window=14).rsi()
    except Exception:
        feat_tmp["rsi_14"] = np.nan
    try:
        macd = ta.trend.MACD(price_series)
        feat_tmp["macd"] = macd.macd()
        feat_tmp["macd_signal"] = macd.macd_signal()
    except Exception:
        feat_tmp["macd"] = np.nan
        feat_tmp["macd_signal"] = np.nan

    for w in [5, 10, 20]:
        feat_tmp[f"sma_{w}"] = price_series.rolling(w).mean()
        feat_tmp[f"ema_{w}"] = price_series.ewm(span=w, adjust=False).mean()

    # Volume features
    if selected_ticker in data_dict and "Volume" in data_dict[selected_ticker].columns:
        vol_series = data_dict[selected_ticker].reindex(price_series.index)["Volume"].ffill()
        feat_tmp["vol_change_1h"] = vol_series.pct_change()
        feat_tmp["vol_ma_24h"] = vol_series.rolling(24).mean()

    # Cross-asset returns
    cross_assets = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "NVDA", "JPM", "JNJ", "XOM", "CAT", "BA", "META"]
    for asset in cross_assets:
        if asset in aligned_ffill.columns:
            feat_tmp[f"{asset}_ret_1h"] = aligned_ffill[asset].pct_change()

    # Calendar features
    feat_tmp["hour"] = feat_tmp.index.hour
    feat_tmp["day_of_week"] = feat_tmp.index.dayofweek

    # Drop rows with NaNs in any feature column
    drop_cols = [col for col in feat_tmp.columns if col not in ["datetime", "ticker"]]
    feat_tmp = feat_tmp.dropna(subset=drop_cols)

    # Enforce column order (from your notebook's X.columns)
    missing = set(feature_columns) - set(feat_tmp.columns)
    if missing:
        st.error(f"Missing columns in features: {missing}")
        return pd.DataFrame()
    feat_tmp = feat_tmp.loc[:, feature_columns]
    return feat_tmp

def create_sequences(X, seq_len=128):
    X_seq = []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
    return np.array(X_seq)

def predict_with_models(features, current_price, scaler, cnn_model, lstm_model, xgb_model, meta_model):
    try:
        # Debug print for troubleshooting
        print("Features columns at prediction:", features.columns.tolist())
        print("Expected feature_columns:", feature_columns)
        print("Feature DataFrame shape:", features.shape)
        print(features.dtypes)
        missing = set(feature_columns) - set(features.columns)
        extra = set(features.columns) - set(feature_columns)
        print("Missing columns:", missing)
        print("Extra columns:", extra)
        features = features.loc[:, feature_columns]
        features_scaled = scaler.transform(features)
        X_seq = create_sequences(features_scaled)
        if len(X_seq) == 0:
            raise ValueError("Not enough data for sequence prediction.")
        recent_sequence = X_seq[-1].reshape(1, 128, features_scaled.shape[1])
        cnn_pred = cnn_model.predict(recent_sequence)[0][0]
        lstm_pred = lstm_model.predict(recent_sequence)[0][0]
        xgb_features = features_scaled[-1].reshape(1, -1)
        xgb_pred = xgb_model.predict(xgb_features)[0]
        meta_features = np.array([[cnn_pred, lstm_pred, xgb_pred]])
        meta_features = np.column_stack([
            meta_features, 
            meta_features.mean(axis=1),
            meta_features.std(axis=1),
            meta_features.max(axis=1),
            meta_features.min(axis=1)
        ])
        final_pred = meta_model.predict(meta_features)[0]
        predicted_price = current_price * (1 + final_pred)
        pred_change_pct = final_pred * 100
        votes = {
            "CNN": "UP" if cnn_pred > current_price else "DOWN",
            "LSTM": "UP" if lstm_pred > current_price else "DOWN",
            "XGBoost": "UP" if xgb_pred > current_price else "DOWN"
        }
        agreement = sum(1 for v in votes.values() if v == ("UP" if pred_change_pct > 0 else "DOWN")) / 3
        confidence = min(95, agreement * 100 + min(20, abs(pred_change_pct) * 2))
        return predicted_price, pred_change_pct, votes, confidence
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None, None

def update_prediction_results():
    if not st.session_state.prediction_log:
        return
    updated_log = []
    correct_predictions = 0
    total_return = 0
    for prediction in st.session_state.prediction_log:
        if 'actual_price' not in prediction:
            ticker = prediction['ticker']
            prediction_time = datetime.strptime(prediction['timestamp'], "%Y-%m-%d %H:%M")
            target_time = prediction_time + timedelta(hours=1)
            try:
                stock_data = fetch_stock_data(ticker, period="2d", interval="60m")
                time_diff = abs(stock_data.index - target_time)
                closest_idx = time_diff.argmin()
                actual_price = stock_data.iloc[closest_idx]['Close']
                predicted_direction = "UP" if prediction['predicted_price'] > prediction['current_price'] else "DOWN"
                actual_direction = "UP" if actual_price > prediction['current_price'] else "DOWN"
                correct = predicted_direction == actual_direction
                return_pct = (actual_price - prediction['current_price']) / prediction['current_price'] * 100
                prediction['actual_price'] = actual_price
                prediction['actual_direction'] = actual_direction
                prediction['correct'] = correct
                prediction['return_pct'] = return_pct
                if correct:
                    correct_predictions += 1
                total_return += return_pct
            except Exception as e:
                st.error(f"Error updating prediction for {ticker}: {str(e)}")
        updated_log.append(prediction)
    st.session_state.prediction_log = updated_log
    if st.session_state.prediction_log:
        st.session_state.performance_metrics['total_predictions'] = len(st.session_state.prediction_log)
        st.session_state.performance_metrics['correct_predictions'] = correct_predictions
        st.session_state.performance_metrics['accuracy'] = correct_predictions / len(st.session_state.prediction_log) * 100
        st.session_state.performance_metrics['total_return'] = total_return
        st.session_state.performance_metrics['avg_return'] = total_return / len(st.session_state.prediction_log)

if cnn_model and lstm_model and xgb_model and meta_model and scaler:
    if update_predictions:
        with st.spinner('Updating prediction results...'):
            update_prediction_results()
    tickers_from_notebook = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "NVDA", "JPM", "JNJ", "XOM", "CAT", "BA", "META"]
    aligned_ffill, all_tickers_data = get_aligned_ffill(tickers_from_notebook, days_history)
    if aligned_ffill is None:
        st.error("Not enough data for selected ticker. Try increasing days of history.")
    else:
        stock_data = all_tickers_data[selected_ticker]
        if stock_data is not None and not stock_data.empty:
            latest_data = stock_data.iloc[-1]
            current_price = latest_data['Close']
            current_time = latest_data.name
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            with col2:
                prev_close = stock_data.iloc[-2]['Close'] if len(stock_data) > 1 else current_price
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100
                st.metric("Change", f"{change:.2f}", f"{change_pct:.2f}%")
            with col3:
                st.metric("Last Updated", current_time.strftime("%Y-%m-%d %H:%M"))
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.1, 
                               subplot_titles=(f'{selected_ticker} Price', 'Volume'),
                               row_width=[0.2, 0.7])
            fig.add_trace(go.Candlestick(x=stock_data.index,
                                         open=stock_data['Open'],
                                         high=stock_data['High'],
                                         low=stock_data['Low'],
                                         close=stock_data['Close'],
                                         name="Price"), row=1, col=1)
            colors = ['red' if row['Open'] > row['Close'] else 'green' 
                      for _, row in stock_data.iterrows()]
            fig.add_trace(go.Bar(x=stock_data.index,
                                 y=stock_data['Volume'],
                                 marker_color=colors,
                                 name="Volume"), row=2, col=1)
            fig.add_vline(x=current_time, line_dash="dash", line_color="white")
            fig.update_layout(height=600, showlegend=False, 
                             xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            if run_prediction:
                with st.spinner('Generating prediction...'):
                    features = create_features_for_app(selected_ticker, aligned_ffill, all_tickers_data)
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
                                st.metric("Predicted Price", f"${predicted_price:.2f}", 
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
                            prediction_time = datetime.now()
                            log_entry = {
                                "timestamp": prediction_time.strftime("%Y-%m-%d %H:%M"),
                                "ticker": selected_ticker,
                                "current_price": current_price,
                                "predicted_price": predicted_price,
                                "predicted_change": pred_change_pct,
                                "confidence": confidence,
                                "signal": "BUY" if pred_change_pct > 0.5 else "SELL" if pred_change_pct < -0.5 else "HOLD"
                            }
                            st.session_state.prediction_log.append(log_entry)
                            st.info("Prediction logged. Use the 'Update Prediction Results' button in the sidebar to check actual results after market hours.")
            if st.session_state.prediction_log:
                st.markdown("## Prediction History")
                log_df = pd.DataFrame(st.session_state.prediction_log)
                if 'correct' in log_df.columns:
                    accuracy = st.session_state.performance_metrics['accuracy']
                    total_return = st.session_state.performance_metrics['total_return']
                    avg_return = st.session_state.performance_metrics['avg_return']
                    st.markdown("### Real-Time Performance")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Predictions", st.session_state.performance_metrics['total_predictions'])
                    with col2:
                        st.metric("Accuracy", f"{accuracy:.1f}%")
                    with col3:
                        st.metric("Total Return", f"{total_return:.2f}%")
                    with col4:
                        st.metric("Avg Return", f"{avg_return:.2f}%")
                st.dataframe(log_df, use_container_width=True)
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
else:
    st.warning("""
    Models failed to load. Please ensure you have the following files in the working directory:
    - cnn_model.pkl
    - lstm_model.pkl
    - xgb_model.pkl
    - meta_learner.pkl
    - scaler.pkl
    """)
