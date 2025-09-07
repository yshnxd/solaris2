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

# Set page configuration
st.set_page_config(
    page_title="SOLARIS: AI Stock Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #7f7f7f;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .positive {
        color: #2ecc71;
    }
    .negative {
        color: #e74c3c;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .log-table {
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# App title and introduction
st.markdown('<h1 class="main-header">SOLARIS</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="sub-header">A Machine Learning Based Hourly Stock Prediction Tool</h2>', unsafe_allow_html=True)

st.write("""
This application combines Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM) networks, 
and XGBoost with a Meta Learner Ensemble for short-term stock market predictions. 
*Note: This is for educational and research purposes only - not financial advice.*
""")

# Initialize session state for prediction log
if 'prediction_log' not in st.session_state:
    st.session_state.prediction_log = []
if 'performance_metrics' not in st.session_state:
    st.session_state.performance_metrics = {
        'total_predictions': 0,
        'correct_predictions': 0,
        'total_return': 0,
        'accuracy': 0
    }

# Load models (with caching)
@st.cache_resource
def load_models():
    """Load all trained models"""
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

# Sidebar for user input
st.sidebar.header("Configuration")
selected_ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL").upper()
days_history = st.sidebar.text_input("Days of Historical Data (90-729)", value="180")

# Validate days input
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

# Function to fetch stock data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(ticker, period="30d", interval="60m"):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            st.error(f"No data found for ticker {ticker}")
            return None
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

# Function to create features (FIXED VERSION)
def create_features(df, ticker, all_tickers_data=None):
    """Create features for prediction - FIXED to match training preprocessing"""
    feat_tmp = pd.DataFrame(index=df.index)
    price_series = df['Close']
    
    # Calculate returns with forward fill to preserve data
    for lag in [1, 3, 6, 12, 24]:
        feat_tmp[f"ret_{lag}h"] = price_series.pct_change(lag).ffill()
    
    # Calculate volatility with forward fill
    for window in [6, 12, 24]:
        feat_tmp[f"vol_{window}h"] = price_series.pct_change().rolling(window).std().ffill()
    
    # Technical indicators with error handling
    try:
        feat_tmp["rsi_14"] = ta.momentum.RSIIndicator(price_series, window=14).rsi().ffill()
    except Exception:
        feat_tmp["rsi_14"] = np.nan
        
    try:
        macd = ta.trend.MACD(price_series)
        feat_tmp["macd"] = macd.macd().ffill()
        feat_tmp["macd_signal"] = macd.macd_signal().ffill()
    except Exception:
        feat_tmp["macd"] = np.nan
        feat_tmp["macd_signal"] = np.nan
    
    # Moving averages with forward fill
    for w in [5, 10, 20]:
        feat_tmp[f"sma_{w}"] = price_series.rolling(w).mean().ffill()
        feat_tmp[f"ema_{w}"] = price_series.ewm(span=w, adjust=False).mean().ffill()
    
    # Volume features with forward fill
    if "Volume" in df.columns:
        vol_series = df["Volume"].ffill()
        feat_tmp["vol_change_1h"] = vol_series.pct_change().ffill()
        feat_tmp["vol_ma_24h"] = vol_series.rolling(24).mean().ffill()
    
    # Cross-asset returns with forward fill
    tickers_from_notebook = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "NVDA", "JPM", "JNJ", "XOM", "CAT", "BA", "META"]
    
    for asset in tickers_from_notebook:
        if asset != ticker and all_tickers_data is not None and asset in all_tickers_data:
            # Use the aligned_close approach from notebook
            feat_tmp[f"{asset}_ret_1h"] = all_tickers_data[asset]['Close'].pct_change().ffill()
        else:
            feat_tmp[f"{asset}_ret_1h"] = np.nan
    
    # Calendar features
    feat_tmp["hour"] = feat_tmp.index.hour
    feat_tmp["day_of_week"] = feat_tmp.index.dayofweek
    
    # Forward fill all columns to handle NaNs
    feat_tmp = feat_tmp.ffill()
    
    # Drop any remaining rows with NaNs
    feat_tmp = feat_tmp.dropna()
    
    return feat_tmp

# Function to prepare sequence data (FIXED to handle edge cases)
def create_sequences(X, seq_len=128):
    """Convert tabular data into sequences for CNN/LSTM - FIXED"""
    X_seq = []
    start_idx = max(0, len(X) - seq_len)  # Ensure we have enough data
    for i in range(start_idx, len(X) - seq_len + 1):
        X_seq.append(X[i:i+seq_len])
    return np.array(X_seq)

# Function to make predictions using the actual models (FIXED)
def predict_with_models(features, current_price, scaler, cnn_model, lstm_model, xgb_model, meta_model):
    """Make predictions using the actual trained models - FIXED"""
    try:
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Prepare data for CNN and LSTM (sequence data)
        X_seq = create_sequences(features_scaled)
        
        if len(X_seq) == 0:
            st.error("Not enough data to create sequences after scaling")
            return None, None, None, None
            
        # Get the most recent sequence
        recent_sequence = X_seq[-1].reshape(1, 128, -1)
        
        # Get predictions from each model
        cnn_pred = cnn_model.predict(recent_sequence, verbose=0)[0][0]
        lstm_pred = lstm_model.predict(recent_sequence, verbose=0)[0][0]
        
        # For XGBoost, use the most recent features (not a sequence)
        xgb_features = features_scaled[-1].reshape(1, -1)
        xgb_pred = xgb_model.predict(xgb_features)[0]
        
        # Prepare meta features (exactly as in notebook)
        meta_features = np.array([[cnn_pred, lstm_pred, xgb_pred]])
        meta_features = np.column_stack([
            meta_features, 
            meta_features.mean(axis=1),
            meta_features.std(axis=1),
            meta_features.max(axis=1),
            meta_features.min(axis=1)
        ])
        
        # Get final prediction from meta learner
        final_pred = meta_model.predict(meta_features)[0]
        
        # Calculate percentage change
        pred_change_pct = (final_pred / current_price - 1) * 100
        
        # Determine model votes
        votes = {
            "CNN": "UP" if cnn_pred > current_price else "DOWN",
            "LSTM": "UP" if lstm_pred > current_price else "DOWN",
            "XGBoost": "UP" if xgb_pred > current_price else "DOWN"
        }
        
        # Calculate confidence based on agreement
        agreement = sum(1 for v in votes.values() if v == ("UP" if pred_change_pct > 0 else "DOWN")) / 3
        confidence = min(95, agreement * 100 + min(20, abs(pred_change_pct) * 2))
        
        return final_pred, pred_change_pct, votes, confidence
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None

# Function to update prediction results with actual prices
def update_prediction_results():
    """Update prediction log with actual prices and calculate accuracy"""
    if not st.session_state.prediction_log:
        return
    
    updated_log = []
    correct_predictions = 0
    total_return = 0
    
    for prediction in st.session_state.prediction_log:
        # Check if we already have the actual price
        if 'actual_price' not in prediction:
            # Fetch the actual price for the prediction time + 1 hour
            ticker = prediction['ticker']
            prediction_time = datetime.strptime(prediction['timestamp'], "%Y-%m-%d %H:%M")
            target_time = prediction_time + timedelta(hours=1)
            
            # Fetch data for the target time
            try:
                # Get data for the day of the prediction
                stock_data = fetch_stock_data(ticker, period="2d", interval="60m")
                
                # Find the closest time to our target
                time_diff = abs(stock_data.index - target_time)
                closest_idx = time_diff.argmin()
                actual_price = stock_data.iloc[closest_idx]['Close']
                
                # Calculate if prediction was correct
                predicted_direction = "UP" if prediction['predicted_price'] > prediction['current_price'] else "DOWN"
                actual_direction = "UP" if actual_price > prediction['current_price'] else "DOWN"
                correct = predicted_direction == actual_direction
                
                # Calculate return
                return_pct = (actual_price - prediction['current_price']) / prediction['current_price'] * 100
                
                # Update prediction with actual data
                prediction['actual_price'] = actual_price
                prediction['actual_direction'] = actual_direction
                prediction['correct'] = correct
                prediction['return_pct'] = return_pct
                
                # Update metrics
                if correct:
                    correct_predictions += 1
                total_return += return_pct
                
            except Exception as e:
                st.error(f"Error updating prediction for {ticker}: {str(e)}")
        
        updated_log.append(prediction)
    
    # Update the log
    st.session_state.prediction_log = updated_log
    
    # Update performance metrics
    if st.session_state.prediction_log:
        st.session_state.performance_metrics['total_predictions'] = len(st.session_state.prediction_log)
        st.session_state.performance_metrics['correct_predictions'] = correct_predictions
        st.session_state.performance_metrics['accuracy'] = correct_predictions / len(st.session_state.prediction_log) * 100
        st.session_state.performance_metrics['total_return'] = total_return
        st.session_state.performance_metrics['avg_return'] = total_return / len(st.session_state.prediction_log)

# Main app
if cnn_model and lstm_model and xgb_model and meta_model and scaler:
    # Update prediction results if requested
    if update_predictions:
        with st.spinner('Updating prediction results...'):
            update_prediction_results()
    
    # Fetch data for selected ticker
    stock_data = fetch_stock_data(selected_ticker, period=f"{days_history}d", interval="60m")
    
    if stock_data is not None and not stock_data.empty:
        # Get the latest data point
        latest_data = stock_data.iloc[-1]
        current_price = latest_data['Close']
        current_time = latest_data.name
        
        # Display current price and info
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
        
        # Create price chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.1, 
                           subplot_titles=(f'{selected_ticker} Price', 'Volume'),
                           row_width=[0.2, 0.7])
        
        # Price trace
        fig.add_trace(go.Candlestick(x=stock_data.index,
                                    open=stock_data['Open'],
                                    high=stock_data['High'],
                                    low=stock_data['Low'],
                                    close=stock_data['Close'],
                                    name="Price"), row=1, col=1)
        
        # Volume trace
        colors = ['red' if row['Open'] > row['Close'] else 'green' 
                  for _, row in stock_data.iterrows()]
        fig.add_trace(go.Bar(x=stock_data.index,
                            y=stock_data['Volume'],
                            marker_color=colors,
                            name="Volume"), row=2, col=1)
        
        # Add vertical line for current time
        fig.add_vline(x=current_time, line_dash="dash", line_color="white")
        
        fig.update_layout(height=600, showlegend=False, 
                         xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Run prediction if requested
        if run_prediction:
            with st.spinner('Generating prediction...'):
                # Fetch data for all tickers used in cross-asset features
                tickers_from_notebook = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "NVDA", "JPM", "JNJ", "XOM", "CAT", "BA", "META"]
                all_tickers_data = {}
                
                for ticker_name in tickers_from_notebook:
                    if ticker_name != selected_ticker:
                        ticker_data = fetch_stock_data(ticker_name, period=f"{days_history}d", interval="60m")
                        if ticker_data is not None:
                            all_tickers_data[ticker_name] = ticker_data
                
                # Create features (exactly as in notebook)
                features = create_features(stock_data, selected_ticker, all_tickers_data)
                
                # Check if we have enough data
                if len(features) < 128:
                    st.error(f"Not enough historical data to generate prediction. Need at least 128 hours of data, but only have {len(features)}.")
                else:
                    # Get prediction using the actual models
                    predicted_price, pred_change_pct, votes, confidence = predict_with_models(
                        features, current_price, scaler, cnn_model, lstm_model, xgb_model, meta_model
                    )
                    
                    if predicted_price is not None:
                        # Determine model votes
                        vote_display = {
                            "CNN": "UP" if votes["CNN"] == "UP" else "DOWN",
                            "LSTM": "UP" if votes["LSTM"] == "UP" else "DOWN",
                            "XGBoost": "UP" if votes["XGBoost"] == "UP" else "DOWN"
                        }
                        
                        # Display prediction results
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
                            # Trading suggestion
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
                        
                        # Log the prediction
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
                        
                        # Show update reminder
                        st.info("Prediction logged. Use the 'Update Prediction Results' button in the sidebar to check actual results after market hours.")
        
        # Display prediction log
        if st.session_state.prediction_log:
            st.markdown("## Prediction History")
            
            # Convert to DataFrame for display
            log_df = pd.DataFrame(st.session_state.prediction_log)
            
            # Add result columns if available
            if 'correct' in log_df.columns:
                # Calculate accuracy metrics
                accuracy = st.session_state.performance_metrics['accuracy']
                total_return = st.session_state.performance_metrics['total_return']
                avg_return = st.session_state.performance_metrics['avg_return']
                
                # Display performance metrics
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
    
    # Model explanation section
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
        
        # Feature importance (from your notebook)
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
        
        # Backtest performance (from your actual results)
        st.markdown("### Historical Backtest Performance")
        st.markdown("""
        Based on extensive backtesting, the model has achieved:
        
        - **Total Return**: 239.97%
        - **Start Capital**: $10,000.00
        - **Final Capital**: $33,997.15
        - **Total Trades**: 8,944
        - **Avg Hourly PnL**: $2.68 ± $91.17
        """)
        
        # Disclaimer
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
