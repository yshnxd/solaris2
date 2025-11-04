import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import joblib
import ta
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import os
import math
import time
import warnings
warnings.filterwarnings("ignore")

# Basic page config (simple, no custom CSS)
st.set_page_config(page_title="StockWise", page_icon="📈", layout="wide")

# Constants
PREDICTION_HISTORY_CSV = "solaris-data.csv"
show_cols = [
    "prediction_id",
    "timestamp",
    "ticker",
    "current_price",
    "predicted_price",
    "target_time",
    "actual_price",
    "error_pct",
    "error_abs",
    "confidence",
    "signal",
]

original_cross_assets = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "TSLA",
    "NVDA",
    "JPM",
    "JNJ",
    "XOM",
    "CAT",
    "BA",
    "META",
]
dynamic_cross_assets = original_cross_assets.copy()

expected_columns = [
    "ret_1h",
    "ret_3h",
    "ret_6h",
    "ret_12h",
    "ret_24h",
    "vol_6h",
    "vol_12h",
    "vol_24h",
    "rsi_14",
    "macd",
    "macd_signal",
    "sma_5",
    "ema_5",
    "sma_10",
    "ema_10",
    "sma_20",
    "ema_20",
    "vol_change_1h",
    "vol_ma_24h",
    "AAPL_ret_1h",
    "MSFT_ret_1h",
    "AMZN_ret_1h",
    "GOOGL_ret_1h",
    "TSLA_ret_1h",
    "NVDA_ret_1h",
    "JPM_ret_1h",
    "JNJ_ret_1h",
    "XOM_ret_1h",
    "CAT_ret_1h",
    "BA_ret_1h",
    "META_ret_1h",
    "hour",
    "day_of_week",
    "price",
]

SEQ_LEN = 128

# Utilities

def ensure_csv_has_header(csv_path: str, columns: list[str]) -> None:
    if not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0:
        pd.DataFrame(columns=columns).to_csv(csv_path, index=False)


def compute_signal(predicted, actual, buy_threshold_pct: float = 0.25, sell_threshold_pct: float = 0.25) -> str:
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


def generate_suggestion(current_price: float, predicted_price: float, pred_change_pct: float, confidence: float, votes: dict) -> tuple[str, str]:
    """
    Generate trading suggestion and explanation based on prediction results.
    Returns: (suggestion, explanation)
    """
    # Determine suggestion based on predicted change and confidence
    strong_buy_threshold = 0.5  # % change threshold for strong buy
    buy_threshold = 0.1  # % change threshold for buy
    sell_threshold = -0.1  # % change threshold for sell
    strong_sell_threshold = -0.5  # % change threshold for strong sell
    high_confidence = 70  # Confidence threshold for high confidence
    
    pred_change = pred_change_pct
    
    # Count model votes
    up_votes = sum(1 for v in votes.values() if v == "UP")
    down_votes = sum(1 for v in votes.values() if v == "DOWN")
    
    # Generate suggestion based on predicted change and confidence
    if pred_change >= strong_buy_threshold and confidence >= high_confidence:
        suggestion = "🟢 STRONG BUY"
        explanation = (
            f"The models predict a strong upward movement of {pred_change:+.2f}% with {confidence:.1f}% confidence. "
            f"All {up_votes} models agree on upward direction. The predicted price ${predicted_price:.2f} represents "
            f"a significant increase from the current price ${current_price:.2f}. Consider entering a position "
            f"with appropriate risk management."
        )
    elif pred_change >= buy_threshold and confidence >= high_confidence:
        suggestion = "🟢 BUY"
        explanation = (
            f"The models predict an upward movement of {pred_change:+.2f}% with {confidence:.1f}% confidence. "
            f"{up_votes} out of 3 models agree on upward direction. The predicted price ${predicted_price:.2f} "
            f"suggests a moderate gain from the current price ${current_price:.2f}. This could be a good entry "
            f"point, but monitor the market closely."
        )
    elif pred_change >= buy_threshold:
        suggestion = "🟡 WEAK BUY"
        explanation = (
            f"The models predict a slight upward movement of {pred_change:+.2f}%, but confidence is only "
            f"{confidence:.1f}%. Only {up_votes} out of 3 models agree on upward direction. The predicted price "
            f"${predicted_price:.2f} is slightly higher than the current price of ${current_price:.2f}. Exercise caution and "
            f"consider waiting for stronger confirmation."
        )
    elif pred_change <= strong_sell_threshold and confidence >= high_confidence:
        suggestion = "🔴 STRONG SELL"
        explanation = (
            f"The models predict a strong downward movement of {pred_change:+.2f}% with {confidence:.1f}% confidence. "
            f"All {down_votes} models agree on downward direction. The predicted price ${predicted_price:.2f} represents "
            f"a significant decrease from the current price ${current_price:.2f}. Consider exiting long positions "
            f"or entering short positions with proper risk management."
        )
    elif pred_change <= sell_threshold and confidence >= high_confidence:
        suggestion = "🔴 SELL"
        explanation = (
            f"The models predict a downward movement of {pred_change:+.2f}% with {confidence:.1f}% confidence. "
            f"{down_votes} out of 3 models agree on downward direction. The predicted price ${predicted_price:.2f} "
            f"suggests a moderate decline from the current price ${current_price:.2f}. Consider reducing positions "
            f"or tightening stop-loss orders."
        )
    elif pred_change <= sell_threshold:
        suggestion = "🟡 WEAK SELL"
        explanation = (
            f"The models predict a slight downward movement of {pred_change:+.2f}%, but confidence is only "
            f"{confidence:.1f}%. Only {down_votes} out of 3 models agree on downward direction. The predicted price "
            f"${predicted_price:.2f} is slightly lower than the current price of ${current_price:.2f}. Exercise caution and "
            f"monitor the position closely."
        )
    else:
        suggestion = "⚪ HOLD"
        # Determine consensus message based on model agreement
        if up_votes == 3:
            consensus_msg = f"All {up_votes} models agree on upward direction, but the predicted change is minimal."
            movement_msg = "The minimal predicted movement suggests sideways trading or consolidation."
        elif down_votes == 3:
            consensus_msg = f"All {down_votes} models agree on downward direction, but the predicted change is minimal."
            movement_msg = "The minimal predicted movement suggests sideways trading or consolidation."
        elif up_votes > down_votes:
            consensus_msg = f"Model consensus is mixed: {up_votes} models suggest up, {down_votes} suggest down."
            movement_msg = "This suggests sideways movement or uncertainty."
        elif down_votes > up_votes:
            consensus_msg = f"Model consensus is mixed: {down_votes} models suggest down, {up_votes} suggest up."
            movement_msg = "This suggests sideways movement or uncertainty."
        else:
            consensus_msg = "Model consensus is mixed with no clear direction."
            movement_msg = "This suggests sideways movement or uncertainty."
        
        explanation = (
            f"The models predict minimal movement ({pred_change:+.2f}%) with {confidence:.1f}% confidence. "
            f"The predicted price ${predicted_price:.2f} is close to the current price of ${current_price:.2f}. "
            f"{consensus_msg} {movement_msg} Consider waiting for clearer signals or maintaining current positions."
        )
    
    return suggestion, explanation


def get_next_prediction_id(csv_path: str) -> int:
    try:
        if not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0:
            return 1
        df = pd.read_csv(csv_path, usecols=["prediction_id"]) if "prediction_id" in pd.read_csv(csv_path, nrows=0).columns else pd.read_csv(csv_path)
        if "prediction_id" not in df.columns or df.empty:
            return 1
        # Coerce to numeric, ignore errors
        ids = pd.to_numeric(df["prediction_id"], errors="coerce").fillna(0).astype(int)
        return int(ids.max()) + 1 if len(ids) > 0 else 1
    except Exception:
        return 1


def fetch_history_with_retry(ticker: str, period: str, interval: str, rate_limit_sec: float = 1.0, max_attempts: int = 3) -> pd.DataFrame | None:
    """
    Fetch Yahoo Finance data safely with retries.
    Returns None if all attempts fail.
    """
    for attempt in range(max_attempts):
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period=period, interval=interval, actions=False, auto_adjust=False)
            if hist.empty or not isinstance(hist.index, pd.DatetimeIndex):
                raise ValueError("Empty or invalid data")
            if hist.index.tz is None:
                hist.index = hist.index.tz_localize("UTC")
            else:
                hist.index = hist.index.tz_convert("UTC")
            return hist
        except Exception as e:
            if attempt == max_attempts - 1:
                # Last attempt failed - log but don't raise
                pass
            time.sleep(rate_limit_sec * 2)
    return None


def batch_evaluate_predictions_csv(csv_path: str, rate_limit_sec: float = 1.0, hour_fetch_period: str = "730d", buy_threshold_pct: float = 1.0, sell_threshold_pct: float = 1.0) -> pd.DataFrame:
    """
    TIMESTAMP LOGIC: This function evaluates predictions by filling in actual prices.
    It preserves original timestamp strings while using datetime objects for time comparisons.
    
    Uses improved evaluation logic:
    - Only uses past bars (no future leakage)
    - Retries with error handling
    - Prefers hourly data, falls back to daily
    """
    ensure_csv_has_header(csv_path, show_cols)
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        if df.empty:
            return pd.DataFrame(columns=show_cols)
    except (pd.errors.EmptyDataError, FileNotFoundError, Exception):
        return pd.DataFrame(columns=show_cols)

    # TIMESTAMP LOGIC: Preserve original timestamp strings before converting to datetime
    # We need datetime objects for time comparisons, but must restore strings before saving
    timestamp_strings = pd.Series(dtype=object)
    target_time_strings = pd.Series(dtype=object)
    if "timestamp" in df.columns:
        # Save original string values before converting to datetime
        timestamp_strings = df["timestamp"].copy().astype(str)
    if "target_time" in df.columns:
        # Save original string values before converting to datetime
        target_time_strings = df["target_time"].copy().astype(str)
    
    # Convert to datetime for time-based operations (comparisons, floor, normalize)
    for col in ["timestamp", "target_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # Ensure necessary columns exist
    for col in ["actual_price", "error_pct", "error_abs", "signal"]:
        if col not in df.columns:
            df[col] = np.nan
    df["signal"] = df["signal"].fillna("HOLD")

    # Group by ticker
    ticker_groups: dict[str, list[int]] = {}
    for idx, row in df.iterrows():
        t = str(row.get("ticker", "")).strip()
        if not t or pd.isna(row.get("target_time")):
            continue
        ticker_groups.setdefault(t, []).append(idx)

    # Main evaluation loop
    for j, (ticker, indices) in enumerate(ticker_groups.items(), start=1):
        # Fetch history with retries
        hist_hour = fetch_history_with_retry(ticker, hour_fetch_period, "1h", rate_limit_sec)
        hist_day = fetch_history_with_retry(ticker, "365d", "1d", rate_limit_sec)

        if (hist_hour is None or hist_hour.empty) and (hist_day is None or hist_day.empty):
            continue

        for i in indices:
            t_utc = df.at[i, "target_time"]
            if pd.isna(t_utc):
                continue
            if not pd.isna(df.at[i, "actual_price"]):
                continue

            close_val = np.nan
            filled = False
            reason = ""

            # Prefer hourly data - only use past bars (no future leakage)
            if hist_hour is not None and not hist_hour.empty:
                before = hist_hour.loc[hist_hour.index <= t_utc.floor("H")]
                if not before.empty:
                    close_val = float(before.iloc[-1]["Close"])
                    filled = True
                    reason = "hourly <= target"

            # Daily fallback
            if not filled and hist_day is not None and not hist_day.empty:
                before_day = hist_day.loc[hist_day.index <= t_utc.normalize()]
                if before_day.empty:
                    before_day = hist_day
                close_val = float(before_day.iloc[-1]["Close"])
                filled = True
                reason = "daily fallback"

            if filled:
                try:
                    predicted = float(df.at[i, "predicted_price"])
                    if not math.isnan(predicted) and close_val != 0:
                        # Error calculation: (predicted - actual) / actual * 100
                        df.at[i, "error_pct"] = (predicted - close_val) / close_val * 100.0
                        df.at[i, "error_abs"] = abs(predicted - close_val)
                except Exception:
                    pass
                df.at[i, "actual_price"] = close_val
                df.at[i, "signal"] = compute_signal(df.at[i, "predicted_price"], close_val, buy_threshold_pct, sell_threshold_pct)

        time.sleep(rate_limit_sec)

    # TIMESTAMP LOGIC: Restore original timestamp strings before saving
    # Replace datetime columns with original string values to preserve format
    if "timestamp" in df.columns and len(timestamp_strings) > 0:
        # Restore original strings - align by index
        df["timestamp"] = timestamp_strings.reindex(df.index).fillna(df["timestamp"].astype(str))
    
    if "target_time" in df.columns and len(target_time_strings) > 0:
        # Restore original strings - align by index
        df["target_time"] = target_time_strings.reindex(df.index).fillna(df["target_time"].astype(str))
    
    # TIMESTAMP LOGIC: Final validation - ensure all timestamp columns are strings before saving
    # Convert any datetime objects that couldn't be restored to strings
    for col in ["timestamp", "target_time"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: x if isinstance(x, str) and str(x).strip() not in ['', 'nan', 'NaT']
                else (x.strftime('%Y-%m-%d %H:%M:%S') if hasattr(x, 'strftime') and pd.notna(x) else (str(x) if pd.notna(x) else ''))
            )
    
    df.to_csv(csv_path, index=False)
    return df


@st.cache_resource
def load_models():
    try:
        cnn_model = joblib.load("cnn_model.pkl")
        lstm_model = joblib.load("lstm_model.pkl")
        xgb_model = joblib.load("xgb_model.pkl")
        meta_model = joblib.load("meta_learner.pkl")
        scaler = joblib.load("scaler.pkl")
        return cnn_model, lstm_model, xgb_model, meta_model, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error("Please ensure all model files are in the working directory.")
        return None, None, None, None, None


cnn_model, lstm_model, xgb_model, meta_model, scaler = load_models()


@st.cache_data(ttl=300)
def fetch_stock_data(ticker: str, period: str = "729d", interval: str = "60m") -> pd.DataFrame | None:
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return None


def fetch_cross_assets(selected_index: pd.DatetimeIndex, days_history: int, selected_ticker: str) -> dict[str, pd.Series]:
    global dynamic_cross_assets
    if selected_ticker in dynamic_cross_assets:
        dynamic_cross_assets = [a for a in original_cross_assets if a != selected_ticker]
    else:
        dynamic_cross_assets = original_cross_assets[:11]

    cross_data: dict[str, pd.Series] = {}
    for asset in dynamic_cross_assets:
        df = fetch_stock_data(asset, period=f"{days_history}d", interval="60m")
        if df is not None and not df.empty:
            cross_data[asset] = df.reindex(selected_index)["Close"].ffill()
        else:
            cross_data[asset] = pd.Series(0, index=selected_index)
    return cross_data


def create_features_for_app(selected_ticker: str, ticker_data: pd.DataFrame, cross_data: dict[str, pd.Series]) -> pd.DataFrame:
    price_series = ticker_data["Close"]
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
        vol_series = ticker_data["Volume"].ffill()
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


def predict_with_models(
    features: pd.DataFrame,
    current_price: float,
    scaler: StandardScaler,
    cnn_model,
    lstm_model,
    xgb_model,
    meta_model,
):
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
    except Exception:
        pred_cnn = 0.0

    try:
        pred_lstm = lstm_model.predict(X_seq, verbose=0)
        pred_lstm = float(pred_lstm[0][0])
    except Exception:
        pred_lstm = 0.0

    try:
        pred_xgb = xgb_model.predict(X_tab)
        pred_xgb = float(pred_xgb[0])
    except Exception:
        pred_xgb = 0.0

    meta_feats = np.array([pred_cnn, pred_lstm, pred_xgb])
    meta_stats = np.array([meta_feats.mean(), meta_feats.std(), meta_feats.max(), meta_feats.min()])
    meta_input = np.concatenate([meta_feats, meta_stats]).reshape(1, -1)

    try:
        pred_meta = meta_model.predict(meta_input)
        pred_meta = float(pred_meta[0])
    except Exception:
        pred_meta = np.mean([pred_cnn, pred_lstm, pred_xgb])

    predicted_price = current_price * (1 + pred_meta)
    pred_change_pct = pred_meta * 100

    votes = {
        "CNN": "UP" if pred_cnn > 0 else "DOWN",
        "LSTM": "UP" if pred_lstm > 0 else "DOWN",
        "XGBoost": "UP" if pred_xgb > 0 else "DOWN",
    }

    up_count = sum(v > 0 for v in [pred_cnn, pred_lstm, pred_xgb])
    down_count = sum(v < 0 for v in [pred_cnn, pred_lstm, pred_xgb])
    confidence = (max(up_count, down_count) / 3) * 100

    return predicted_price, pred_change_pct, votes, confidence


# Sidebar controls
st.sidebar.header("Controls")
selected_ticker = st.sidebar.text_input("Ticker", "AAPL").upper()
days_history = st.sidebar.slider("Historical days", min_value=30, max_value=729, value=90)
run_prediction = st.sidebar.button("Generate Prediction")
evaluate_results = st.sidebar.button("Evaluate History")
run_watchlist = st.sidebar.button("Watchlist Analysis")

st.sidebar.subheader("Chart Overlays")
overlay_options = st.sidebar.multiselect(
    "Moving Averages", ["SMA 5", "SMA 10", "SMA 20", "EMA 5", "EMA 10", "EMA 20"], default=["SMA 20"]
)

st.sidebar.subheader("Indicators")
show_rsi = st.sidebar.checkbox("RSI (14)", value=False)
show_macd = st.sidebar.checkbox("MACD", value=False)

# Tabs
tab1, tab2, tab3 = st.tabs(["Dashboard", "Prediction History", "Theory"])

# Global actions
if evaluate_results:
    if os.path.exists(PREDICTION_HISTORY_CSV):
        with st.spinner("Evaluating prediction history..."):
            try:
                _ = batch_evaluate_predictions_csv(PREDICTION_HISTORY_CSV)
                st.success("Prediction history updated with actual prices.")
                st.rerun()
            except Exception as e:
                st.error(f"Error evaluating predictions: {str(e)}")
    else:
        st.info("No prediction history found.")

with tab1:
    st.title("StockWise")
    st.caption("Hybrid Machine Learning Stock Price Prediction")

    # Fetch data
    main_ticker_data = fetch_stock_data(selected_ticker, period=f"{days_history}d", interval="60m")

    if main_ticker_data is None or main_ticker_data.empty or len(main_ticker_data) < SEQ_LEN:
        st.warning("No chart data available. Select a valid ticker and ensure sufficient data.")
    else:
        latest_data = main_ticker_data.iloc[-1]
        current_price = float(latest_data["Close"]) if not pd.isna(latest_data["Close"]) else np.nan

        # Metrics
        cols = st.columns(3)
        with cols[0]:
            st.metric("Current Price", f"${current_price:.2f}" if not pd.isna(current_price) else "N/A")
        with cols[1]:
            prev_close = main_ticker_data.iloc[-2]["Close"] if len(main_ticker_data) > 1 else current_price
            if pd.isna(prev_close) or pd.isna(current_price) or prev_close == 0:
                st.metric("Change (1h)", "N/A")
            else:
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100
                st.metric("Change (1h)", f"{change:+.2f}", f"{change_pct:+.2f}%")
        with cols[2]:
            vol_val = latest_data.get("Volume", np.nan)
            st.metric("Volume", f"{vol_val:,.0f}" if not pd.isna(vol_val) else "N/A")

        # Chart
        st.subheader("Price Chart")
        fig = go.Figure()
        fig.add_candlestick(
            x=main_ticker_data.index,
            open=main_ticker_data["Open"],
            high=main_ticker_data["High"],
            low=main_ticker_data["Low"],
            close=main_ticker_data["Close"],
            name="Price",
        )

        if "SMA 5" in overlay_options:
            fig.add_scatter(x=main_ticker_data.index, y=main_ticker_data["Close"].rolling(5).mean(), name="SMA 5")
        if "SMA 10" in overlay_options:
            fig.add_scatter(x=main_ticker_data.index, y=main_ticker_data["Close"].rolling(10).mean(), name="SMA 10")
        if "SMA 20" in overlay_options:
            fig.add_scatter(x=main_ticker_data.index, y=main_ticker_data["Close"].rolling(20).mean(), name="SMA 20")
        if "EMA 5" in overlay_options:
            fig.add_scatter(x=main_ticker_data.index, y=main_ticker_data["Close"].ewm(span=5).mean(), name="EMA 5")
        if "EMA 10" in overlay_options:
            fig.add_scatter(x=main_ticker_data.index, y=main_ticker_data["Close"].ewm(span=10).mean(), name="EMA 10")
        if "EMA 20" in overlay_options:
            fig.add_scatter(x=main_ticker_data.index, y=main_ticker_data["Close"].ewm(span=20).mean(), name="EMA 20")

        st.plotly_chart(fig, use_container_width=True)

        # Prediction
        st.subheader("ML Prediction")
        trigger = run_prediction or st.button("Generate ML Prediction")
        if trigger:
            with st.spinner("Running models..."):
                try:
                    cross_data = fetch_cross_assets(main_ticker_data.index, days_history, selected_ticker)
                    features = create_features_for_app(selected_ticker, main_ticker_data, cross_data)
                    if features is not None and len(features) >= SEQ_LEN:
                        predicted_price, pred_change_pct, votes, confidence = predict_with_models(
                            features, current_price, scaler, cnn_model, lstm_model, xgb_model, meta_model
                        )
                        if predicted_price is not None:
                            cols2 = st.columns(2)
                            with cols2[0]:
                                st.metric("Predicted Price (1h)", f"${predicted_price:.2f}")
                            with cols2[1]:
                                st.metric("Expected Change", f"{pred_change_pct:+.2f}%")
                            st.text(f"Consensus: CNN {votes['CNN']} | LSTM {votes['LSTM']} | XGBoost {votes['XGBoost']}")
                            st.text(f"Confidence: {confidence:.1f}%")
                            
                            # Generate and display suggestion with explanation
                            suggestion, explanation = generate_suggestion(
                                current_price, predicted_price, pred_change_pct, confidence, votes
                            )
                            st.markdown("---")
                            st.subheader("💡 Trading Suggestion")
                            st.markdown(f"### {suggestion}")
                            st.text(explanation)

                            # TIMESTAMP LOGIC: Capture exact time when prediction is made
                            # Record current time as ISO format string for consistent storage
                            now = datetime.now()
                            target_time = now + timedelta(hours=1)
                            timestamp_str = now.strftime('%Y-%m-%d %H:%M:%S')
                            target_time_str = target_time.strftime('%Y-%m-%d %H:%M:%S')
                            
                            # Display timestamp immediately in UI
                            st.info(f"📅 Prediction created at: {timestamp_str} | Target time: {target_time_str}")
                            
                            # Ensure CSV file exists with proper headers
                            ensure_csv_has_header(PREDICTION_HISTORY_CSV, show_cols)
                            next_id = get_next_prediction_id(PREDICTION_HISTORY_CSV)
                            
                            # TIMESTAMP LOGIC: Create prediction record with explicit timestamps
                            # All timestamp fields are stored as formatted strings
                            prediction_record = {
                                "prediction_id": next_id,
                                "timestamp": timestamp_str,  # Exact time prediction was made
                                "ticker": selected_ticker,
                                "current_price": current_price,
                                "predicted_price": predicted_price,
                                "target_time": target_time_str,  # Time when prediction should be evaluated
                                "actual_price": None,
                                "error_pct": None,
                                "error_abs": None,
                                "confidence": confidence,
                                "signal": "HOLD",
                            }
                            
                            # TIMESTAMP LOGIC: Read existing CSV while preserving timestamp format
                            # Use low_memory=False and explicit dtype to prevent auto-parsing of dates
                            try:
                                df_hist = pd.read_csv(PREDICTION_HISTORY_CSV, low_memory=False)
                                # Ensure timestamp columns are strings (pandas may auto-convert)
                                if 'timestamp' in df_hist.columns:
                                    df_hist['timestamp'] = df_hist['timestamp'].astype(str).replace('nan', '')
                                if 'target_time' in df_hist.columns:
                                    df_hist['target_time'] = df_hist['target_time'].astype(str).replace('nan', '')
                            except Exception as e:
                                # If reading fails, create empty dataframe with correct structure
                                df_hist = pd.DataFrame(columns=show_cols)
                            
                            # TIMESTAMP LOGIC: Add new prediction row with timestamps
                            # Create dataframe from record, ensuring all required columns exist
                            new_row = pd.DataFrame([prediction_record])
                            # Make sure new row has all required columns
                            for col in show_cols:
                                if col not in new_row.columns:
                                    new_row[col] = None
                            new_row = new_row[show_cols]  # Reorder to match show_cols
                            
                            # Concatenate with existing data
                            df_hist = pd.concat([df_hist, new_row], ignore_index=True, sort=False)
                            
                            # TIMESTAMP LOGIC: Final validation - ensure timestamps are preserved as strings
                            # Convert any datetime objects back to strings before saving
                            for col in ['timestamp', 'target_time']:
                                if col in df_hist.columns:
                                    df_hist[col] = df_hist[col].apply(
                                        lambda x: x if isinstance(x, str) and x != 'nan' and x != '' 
                                        else (x.strftime('%Y-%m-%d %H:%M:%S') if hasattr(x, 'strftime') else str(x) if pd.notna(x) else '')
                                    )
                            
                            # Save to CSV - timestamps should now be properly formatted strings
                            df_hist.to_csv(PREDICTION_HISTORY_CSV, index=False)
                            st.success(f"✅ Prediction saved to history (ID {next_id}) at {timestamp_str}")
                        else:
                            st.error("Failed to generate prediction.")
                    else:
                        st.warning("Insufficient data for prediction (need at least 128 points).")
                except Exception as e:
                    st.error(f"Error generating prediction: {str(e)}")

        # Watchlist
        if run_watchlist:
            st.subheader("Watchlist Analysis")
            # Use 12 watchlist stocks
            watchlist_stocks = original_cross_assets[:12]
            progress = st.progress(0)
            status = st.empty()
            results = []

            # Prepare history file and next_id base
            ensure_csv_has_header(PREDICTION_HISTORY_CSV, show_cols)
            next_id = get_next_prediction_id(PREDICTION_HISTORY_CSV)

            for i, stock in enumerate(watchlist_stocks):
                status.text(f"Analyzing {stock} ({i+1}/{len(watchlist_stocks)})...")
                try:
                    stock_data = fetch_stock_data(stock, period=f"{days_history}d", interval="60m")
                    if stock_data is not None and len(stock_data) >= SEQ_LEN:
                        latest = stock_data.iloc[-1]
                        cur_px = float(latest["Close"]) if not pd.isna(latest["Close"]) else np.nan
                        cross_d = fetch_cross_assets(stock_data.index, days_history, stock)
                        feats = create_features_for_app(stock, stock_data, cross_d)
                        if feats is not None and len(feats) >= SEQ_LEN and not pd.isna(cur_px):
                            pred_px, change_pct, votes_w, conf = predict_with_models(
                                feats, cur_px, scaler, cnn_model, lstm_model, xgb_model, meta_model
                            )
                            if pred_px is not None:
                                # Append to results table
                                results.append(
                                    {
                                        "ticker": stock,
                                        "current_price": cur_px,
                                        "predicted_price": pred_px,
                                        "change_pct": change_pct,
                                        "confidence": conf,
                                        "votes": votes_w,
                                    }
                                )
                                # TIMESTAMP LOGIC: Capture exact time when watchlist prediction is made
                                # Record current time for this specific stock prediction
                                now = datetime.now()
                                target_time = now + timedelta(hours=1)
                                timestamp_str = now.strftime('%Y-%m-%d %H:%M:%S')
                                target_time_str = target_time.strftime('%Y-%m-%d %H:%M:%S')
                                
                                # TIMESTAMP LOGIC: Create prediction record with explicit timestamps
                                # All timestamp fields are stored as formatted strings
                                prediction_record = {
                                    "prediction_id": next_id,
                                    "timestamp": timestamp_str,  # Exact time prediction was made
                                    "ticker": stock,
                                    "current_price": cur_px,
                                    "predicted_price": pred_px,
                                    "target_time": target_time_str,  # Time when prediction should be evaluated
                                    "actual_price": None,
                                    "error_pct": None,
                                    "error_abs": None,
                                    "confidence": conf,
                                    "signal": "HOLD",
                                }
                                
                                # TIMESTAMP LOGIC: Read existing CSV while preserving timestamp format
                                # Use low_memory=False to prevent pandas from auto-parsing dates
                                try:
                                    df_hist = pd.read_csv(PREDICTION_HISTORY_CSV, low_memory=False)
                                    # Ensure timestamp columns are strings (pandas may auto-convert)
                                    if 'timestamp' in df_hist.columns:
                                        df_hist['timestamp'] = df_hist['timestamp'].astype(str).replace('nan', '')
                                    if 'target_time' in df_hist.columns:
                                        df_hist['target_time'] = df_hist['target_time'].astype(str).replace('nan', '')
                                except Exception:
                                    df_hist = pd.DataFrame(columns=show_cols)
                                
                                # TIMESTAMP LOGIC: Add new prediction row with timestamps
                                # Create dataframe from record, ensuring all required columns exist
                                new_row = pd.DataFrame([prediction_record])
                                # Make sure new row has all required columns
                                for col in show_cols:
                                    if col not in new_row.columns:
                                        new_row[col] = None
                                new_row = new_row[show_cols]  # Reorder to match show_cols
                                
                                # Concatenate with existing data
                                df_hist = pd.concat([df_hist, new_row], ignore_index=True, sort=False)
                                
                                # TIMESTAMP LOGIC: Final validation - ensure timestamps are preserved as strings
                                # Convert any datetime objects back to strings before saving
                                for col in ['timestamp', 'target_time']:
                                    if col in df_hist.columns:
                                        df_hist[col] = df_hist[col].apply(
                                            lambda x: x if isinstance(x, str) and x != 'nan' and x != '' 
                                            else (x.strftime('%Y-%m-%d %H:%M:%S') if hasattr(x, 'strftime') else str(x) if pd.notna(x) else '')
                                        )
                                
                                # Save to CSV - timestamps should now be properly formatted strings
                                df_hist.to_csv(PREDICTION_HISTORY_CSV, index=False)
                                next_id += 1
                except Exception as e:
                    st.warning(f"Could not analyze {stock}: {str(e)}")
                progress.progress((i + 1) / len(watchlist_stocks))
                time.sleep(0.2)

            status.text("Done.")
            if results:
                results_df = pd.DataFrame(
                    [
                        {
                            "Ticker": r["ticker"],
                            "Current": r["current_price"],
                            "Predicted": r["predicted_price"],
                            "Change %": r["change_pct"],
                            "Confidence %": r["confidence"],
                            "Votes": f"{r['votes']['CNN']}/{r['votes']['LSTM']}/{r['votes']['XGBoost']}",
                        }
                        for r in results
                    ]
                ).sort_values("Confidence %", ascending=False)
                st.dataframe(results_df, use_container_width=True)
                st.success("Watchlist predictions logged to history.")
            else:
                st.info("No predictions could be generated for the watchlist.")

        # Indicators (optional simple plots)
        if show_rsi:
            st.subheader("RSI (14)")
            try:
                rsi_values = ta.momentum.RSIIndicator(main_ticker_data["Close"], window=14).rsi()
                fig_rsi = go.Figure()
                fig_rsi.add_scatter(x=main_ticker_data.index, y=rsi_values, name="RSI")
                fig_rsi.add_hline(y=70, line_dash="dash")
                fig_rsi.add_hline(y=30, line_dash="dash")
                st.plotly_chart(fig_rsi, use_container_width=True)
            except Exception:
                st.info("RSI not available.")

        if show_macd:
            st.subheader("MACD")
            try:
                macd_indicator = ta.trend.MACD(main_ticker_data["Close"])
                fig_macd = go.Figure()
                fig_macd.add_scatter(x=main_ticker_data.index, y=macd_indicator.macd(), name="MACD")
                fig_macd.add_scatter(x=main_ticker_data.index, y=macd_indicator.macd_signal(), name="Signal")
                st.plotly_chart(fig_macd, use_container_width=True)
            except Exception:
                st.info("MACD not available.")

with tab2:
    st.header("Prediction History")
    if os.path.exists(PREDICTION_HISTORY_CSV):
        try:
            history_df = pd.read_csv(PREDICTION_HISTORY_CSV)
            if not history_df.empty:
                # Summary
                total_predictions = len(history_df)
                completed_predictions = history_df[history_df["actual_price"].notna()]
                avg_error_pct = abs(completed_predictions["error_pct"]).mean() if len(completed_predictions) > 0 else 0
                avg_abs_error = completed_predictions["error_abs"].mean() if len(completed_predictions) > 0 else 0
                avg_confidence = history_df["confidence"].mean() if "confidence" in history_df.columns else 0

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Predictions", total_predictions)
                c2.metric("Avg Error %", f"{avg_error_pct:.2f}%")
                c3.metric("Completed", len(completed_predictions))
                c4.metric("Avg Error $", f"${avg_abs_error:.2f}")

                c5, _ = st.columns(2)
                c5.metric("Avg Confidence", f"{avg_confidence:.1f}%")

                colA, _ = st.columns(2)
                with colA:
                    if st.button("Refresh Actual Prices"):
                        with st.spinner("Updating actual prices..."):
                            _ = batch_evaluate_predictions_csv(PREDICTION_HISTORY_CSV)
                            st.success("Actual prices updated.")
                            st.rerun()

                # Full detailed table
                st.subheader("Full Detailed Table")
                df_full = history_df.copy()
                
                # TIMESTAMP LOGIC: Handle timestamp columns for display
                # Try to convert to datetime for better sorting, but preserve original string format
                for col in ["timestamp", "target_time"]:
                    if col in df_full.columns:
                        # First convert any NaN to empty string
                        df_full[col] = df_full[col].fillna('')
                        # Try to convert to datetime for sorting, but keep original format
                        df_full[col + '_sort'] = pd.to_datetime(df_full[col], errors='coerce')
                
                # Sort by timestamp if available
                if "timestamp_sort" in df_full.columns:
                    df_full = df_full.sort_values("timestamp_sort", ascending=False, na_position='last')
                    df_full = df_full.drop(columns=["timestamp_sort"])
                    if "target_time_sort" in df_full.columns:
                        df_full = df_full.drop(columns=["target_time_sort"])
                
                st.dataframe(df_full, use_container_width=True)

                # Simplified table
                st.subheader("Simplified View")
                # TIMESTAMP LOGIC: Simplified view includes timestamps for quick reference
                # Timestamps are displayed as stored strings for clarity
                simplified_cols = [
                    col for col in [
                        "prediction_id",
                        "ticker",
                        "timestamp",      # When prediction was made
                        "target_time",   # When prediction should be evaluated
                        "current_price",
                        "predicted_price",
                        "actual_price",
                    ]
                    if col in df_full.columns
                ]
                df_simple = df_full[simplified_cols].copy()
                # Basic formatting - preserve timestamps as strings, format numeric columns
                for c in ["current_price", "predicted_price", "actual_price"]:
                    if c in df_simple.columns:
                        df_simple[c] = pd.to_numeric(df_simple[c], errors="coerce")
                # Ensure timestamp columns remain as strings for display
                for c in ["timestamp", "target_time"]:
                    if c in df_simple.columns:
                        df_simple[c] = df_simple[c].astype(str).replace('nan', '').replace('NaT', '')
                st.dataframe(df_simple, use_container_width=True)
            else:
                st.info("No prediction history yet.")
        except Exception as e:
            st.error(f"Error loading prediction history: {str(e)}")
    else:
        st.info("No prediction history file found.")

with tab3:
    st.header("Theory")
    
    st.markdown("""
    The StockWise system is built upon the theoretical foundation of time series forecasting, 
    machine learning regression, and ensemble learning. It operates on the principle that stock prices, 
    though highly volatile, exhibit short-term statistical dependencies that can be learned from historical 
    data through proper feature engineering and model design.
    """)
    
    st.markdown("---")
    
    st.subheader("Problem Formulation")
    
    st.markdown("""
    The app's main predictive task is **one-hour-ahead stock price forecasting**. This is formulated as a 
    supervised regression problem where the target variable is the next-hour return, defined as:
    """)
    
    st.latex(r"y_{t+1} = \frac{p_{t+1} - p_t}{p_t}")
    
    st.markdown("""
    where $p_t$ is the stock's closing price at time $t$. Using returns rather than raw prices allows 
    the model to generalize across multiple tickers with varying price scales and to capture relative 
    percentage changes instead of absolute price levels.
    """)
    
    st.markdown("---")
    
    st.subheader("Feature Construction")
    
    st.markdown("""
    StockWise generates predictive features from historical price and volume data. Each feature is designed 
    to represent a specific financial behavior such as trend, momentum, or volatility. All computations are 
    **causal**—meaning each feature at time $t$ only uses data up to time $t$, ensuring no information from 
    the future leaks into the training process.
    """)
    
    st.markdown("#### 1. Lagged Returns")
    st.markdown("Momentum is represented through lagged returns of different horizons:")
    st.latex(r"r_t^{(h)} = \frac{p_t - p_{t-h}}{p_{t-h}}, \quad h \in \{1, 3, 6, 12, 24\}")
    st.markdown("These indicate how much the price has changed in recent hours and serve as short to long-term momentum indicators.")
    
    st.markdown("#### 2. Rolling Volatility")
    st.markdown("Volatility measures the degree of price fluctuations within a specific period:")
    st.latex(r"\sigma_{t,W} = \sqrt{\frac{1}{W} \sum_{i=0}^{W-1} (r_{t-i}^{(1)} - \bar{r}_{t,W})^2}")
    st.markdown("with $\\bar{r}_{t,W}$ being the mean of recent returns over window $W$. This captures the market's recent turbulence or stability.")
    
    st.markdown("#### 3. Relative Strength Index (RSI)")
    st.markdown("RSI gauges the speed and change of price movements using a 14-hour window:")
    st.latex(r"\text{RSI}_t = 100 - \frac{100}{1 + \text{RS}_t}, \quad \text{where } \text{RS}_t = \frac{\text{AvgGain}_t}{\text{AvgLoss}_t}")
    st.markdown("This provides insight into overbought or oversold conditions.")
    
    st.markdown("#### 4. MACD and Signal Line")
    st.markdown("Trend-following behavior is extracted using moving averages:")
    st.latex(r"\text{MACD}_t = \text{EMA}_t^{(12)} - \text{EMA}_t^{(26)}, \quad \text{Signal}_t = \text{EMA}_t^{(9)}(\text{MACD})")
    st.markdown("MACD and its difference from the signal line indicate potential reversals or momentum shifts.")
    
    st.markdown("#### 5. SMA and EMA")
    st.markdown("Trend indicators are further strengthened by simple and exponential moving averages:")
    st.latex(r"\text{SMA}_{t,w} = \frac{1}{w} \sum_{i=0}^{w-1} p_{t-i}")
    st.markdown("These smooth price data and capture directional trends.")
    
    st.markdown("#### 6. Volume Features")
    st.markdown("Volume dynamics are measured through percent changes and rolling averages:")
    st.latex(r"\Delta V_t^{(1)} = \frac{V_t - V_{t-1}}{V_{t-1}}, \quad \bar{V}_{t,24} = \frac{1}{24} \sum_{i=0}^{23} V_{t-i}")
    st.markdown("These help reflect liquidity and market participation intensity.")
    
    st.markdown("#### 7. Cross-Asset Returns")
    st.markdown("Returns of other major tickers are included:")
    st.latex(r"r_t^{(a,1)} = \frac{p_t^{(a)} - p_{t-1}^{(a)}}{p_{t-1}^{(a)}}")
    st.markdown("These cross-asset influences capture broader sector and market sentiment.")
    
    st.markdown("#### 8. Calendar and Time Features")
    st.markdown("Temporal periodicities are represented using cyclical encodings of hour and day:")
    st.latex(r"\text{hour\_sin}_t = \sin\left(\frac{2\pi \cdot \text{hour}_t}{24}\right), \quad \text{hour\_cos}_t = \cos\left(\frac{2\pi \cdot \text{hour}_t}{24}\right)")
    st.markdown("This allows the model to recognize repeating intraday patterns.")
    
    st.markdown("""
    **Normalization:** Before model training, all features are standardized using Z-score normalization, 
    fitted only on the training data to prevent data leakage.
    """)
    
    st.markdown("---")
    
    st.subheader("Model Architecture")
    
    st.markdown("""
    StockWise employs a **hybrid ensemble learning approach** that integrates three distinct predictive 
    models—XGBoost, CNN, and LSTM—combined through a stacking meta-learner. Each component serves a 
    specialized purpose within the forecasting pipeline.
    """)
    
    st.markdown("#### 1. XGBoost (Extreme Gradient Boosting)")
    st.markdown("""
    A tree-based model that captures complex nonlinear relationships among engineered tabular features. 
    It handles missing data effectively, performs implicit feature selection, and is robust against 
    overfitting through regularization. In StockWise, XGBoost is trained to minimize Mean Absolute Error (MAE) 
    using the objective `reg:absoluteerror`.
    """)
    
    st.markdown("#### 2. Convolutional Neural Network (CNN)")
    st.markdown("""
    The CNN analyzes sequences of recent returns and feature values. It acts as a pattern detector, 
    identifying short-term local temporal structures—such as repetitive micro-trends or volatility spikes—that 
    influence future price movements. Through multiple convolutional layers, the CNN automatically learns 
    useful filters and local dependencies that hand-engineered indicators might miss.
    """)
    
    st.markdown("#### 3. Long Short-Term Memory Network (LSTM)")
    st.markdown("""
    The LSTM is a recurrent neural network capable of modeling long-term temporal dependencies. It maintains 
    hidden memory states that learn from extended sequences, making it particularly effective for time series 
    with delayed reactions or momentum that spans many hours. In StockWise, the LSTM processes the same 
    sequential input as the CNN but focuses on broader context and memory of recent trends.
    """)
    
    st.markdown("---")
    
    st.subheader("Stacking Meta-Learner")
    
    st.markdown("""
    Once the three base models (XGBoost, CNN, and LSTM) are trained, their predictions are combined through 
    a **stacking meta-learner**. During training, the meta-learner does not directly access the true labels 
    of the base models' training data; instead, it learns from out-of-fold (OOF) predictions generated through 
    time-based cross-validation. This ensures that the meta-learner only sees base predictions that are not 
    biased by overfitting.
    """)
    
    st.markdown("Mathematically, the final ensemble prediction is:")
    st.latex(r"\hat{y}_t^{(\text{meta})} = w_1 \hat{y}_t^{(\text{XGB})} + w_2 \hat{y}_t^{(\text{CNN})} + w_3 \hat{y}_t^{(\text{LSTM})} + b")
    
    st.markdown("""
    where $w_i$ are weights learned by the meta-learner to minimize the overall error. If the individual 
    model errors are not perfectly correlated, stacking reduces variance and improves generalization.
    """)
    
    st.markdown("""
    The meta-learner used in StockWise is a **HistGradientBoosting Regressor**, chosen for its speed and 
    strong performance on medium-sized tabular datasets. It learns how to balance and trust each base model 
    differently depending on market conditions and stock characteristics.
    """)
    
    st.markdown("---")
    
    st.subheader("System Workflow")
    
    st.markdown("""
    The complete StockWise forecasting process follows these sequential steps:
    
    1. **User Input:** The user selects a stock ticker and prediction horizon (default: one hour ahead).
    
    2. **Data Retrieval:** The system fetches the latest hourly OHLCV data for the selected ticker.
    
    3. **Data Alignment:** Prices are aligned across all tickers; missing values are handled through forward-filling where appropriate.
    
    4. **Feature Engineering:** The app computes all indicators—lagged returns, volatility, RSI, MACD, SMA, EMA, volume features, cross-asset returns, and time-based encodings.
    
    5. **Normalization:** Data are scaled using pre-fitted training scalers.
    
    6. **Base Model Predictions:** Each base model (XGBoost, CNN, LSTM) produces its own one-hour-ahead forecast.
    
    7. **Meta-Learner Prediction:** The out-of-fold-trained meta-learner combines base predictions to produce the final forecasted return.
    
    8. **Output Conversion:** The predicted return is converted into a forecasted price:
    """)
    
    st.latex(r"\hat{p}_{t+1} = p_t \times (1 + \hat{y}_{t+1})")
    
    st.markdown("""
    The app then displays the predicted next-hour price, the direction of movement (up or down), and a 
    confidence indicator derived from ensemble disagreement.
    
    9. **Logging:** All predictions, intermediate outputs, and timestamps are saved to evaluate live accuracy over time.
    """)
    
    st.markdown("---")
    
    st.subheader("Evaluation Metrics")
    
    st.markdown("""
    The app and research paper primarily assess accuracy through **Mean Absolute Error (MAE)** and 
    **Mean Absolute Percentage Error (MAPE)**:
    """)
    
    st.latex(r"\text{MAE} = \frac{1}{N} \sum |y_t - \hat{y}_t|, \quad \text{MAPE} = \frac{100\%}{N} \sum \left|\frac{y_t - \hat{y}_t}{y_t}\right|")
    
    st.markdown("""
    A lower MAE and MAPE indicate higher predictive accuracy. The StockWise hybrid meta-learner achieved a 
    test MAE of **0.003674**, outperforming its base models and achieving an average live absolute price error 
    of **$1.22 (≈0.67%)** across 100 predictions.
    """)
    

