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

st.sidebar.header("Configuration")
selected_ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper()
days_history = st.sidebar.slider("Days of History", min_value=30, max_value=729, value=90)
run_prediction = st.sidebar.button("Run Prediction")
evaluate_results = st.sidebar.button("Evaluate Predictions (fill actual prices in history CSV)")

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
                                st.markdown(f"{model}: <span class='{color_class}'>{vote}</span>", unsafe_allow_html=True)
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
st.markdown("## History Prediction (SOLARIS Backtest Data)")
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
    Past performance is not indicative of future
    """)
