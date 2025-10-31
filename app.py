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
st.set_page_config(page_title="StockWise: Simple", page_icon="📈", layout="wide")

# Constants
PREDICTION_HISTORY_CSV = "solaris-data.csv"
show_cols = [
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


def batch_evaluate_predictions_csv(csv_path: str, rate_limit_sec: float = 0.5, hour_fetch_period: str = "730d") -> pd.DataFrame:
    ensure_csv_has_header(csv_path, show_cols)
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return pd.DataFrame(columns=show_cols)
    except (pd.errors.EmptyDataError, FileNotFoundError, Exception):
        return pd.DataFrame(columns=show_cols)

    for col in ["timestamp", "target_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    for col in ["actual_price", "error_pct", "error_abs", "signal"]:
        if col not in df.columns:
            df[col] = np.nan
    df["signal"] = df["signal"].fillna("HOLD")

    ticker_groups: dict[str, list[int]] = {}
    for idx, row in df.iterrows():
        t = str(row.get("ticker", "")).strip()
        if t == "" or pd.isna(row.get("target_time")):
            continue
        ticker_groups.setdefault(t, []).append(idx)

    for ticker, indices in ticker_groups.items():
        try:
            tk = yf.Ticker(ticker)
            hist_hour = tk.history(period=hour_fetch_period, interval="1h", actions=False, auto_adjust=False)
            if isinstance(hist_hour.index, pd.DatetimeIndex):
                if hist_hour.index.tz is None:
                    hist_hour.index = hist_hour.index.tz_localize("UTC")
                else:
                    hist_hour.index = hist_hour.index.tz_convert("UTC")
        except Exception:
            hist_hour = None

        try:
            hist_day = tk.history(period="365d", interval="1d", actions=False, auto_adjust=False)
            if isinstance(hist_day.index, pd.DatetimeIndex):
                if hist_day.index.tz is None:
                    hist_day.index = hist_day.index.tz_localize("UTC")
                else:
                    hist_day.index = hist_day.index.tz_convert("UTC")
        except Exception:
            hist_day = None

        for i in indices:
            t_utc = df.at[i, "target_time"]
            if pd.isna(t_utc):
                continue
            if not pd.isna(df.at[i, "actual_price"]):
                continue

            filled = False
            if hist_hour is not None and len(hist_hour) > 0:
                target_hour = t_utc.floor("H")
                if target_hour in hist_hour.index:
                    close_val = float(hist_hour.loc[target_hour]["Close"])
                else:
                    diffs = (hist_hour.index - t_utc).total_seconds()
                    best_idx = np.abs(diffs).argmin()
                    close_val = float(hist_hour.iloc[best_idx]["Close"])
                df.at[i, "actual_price"] = close_val
                filled = True
                try:
                    predicted = float(df.at[i, "predicted_price"])
                    if not math.isnan(predicted) and close_val != 0:
                        df.at[i, "error_pct"] = (close_val - predicted) / close_val * 100.0
                        df.at[i, "error_abs"] = abs(close_val - predicted)
                except Exception:
                    pass
                df.at[i, "signal"] = compute_signal(df.at[i, "predicted_price"], close_val)

            if not filled and hist_day is not None and len(hist_day) > 0:
                target_date = t_utc.normalize()
                candidate = hist_day.loc[hist_day.index <= target_date]
                if candidate is None or len(candidate) == 0:
                    candidate = hist_day
                close_val = float(candidate.iloc[-1]["Close"])
                df.at[i, "actual_price"] = close_val
                try:
                    predicted = float(df.at[i, "predicted_price"])
                    if not math.isnan(predicted) and close_val != 0:
                        df.at[i, "error_pct"] = (close_val - predicted) / close_val * 100.0
                        df.at[i, "error_abs"] = abs(close_val - predicted)
                except Exception:
                    pass
                df.at[i, "signal"] = compute_signal(df.at[i, "predicted_price"], close_val)

        time.sleep(rate_limit_sec)

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
evaluate_results = st.sidebar.button("Evaluate History (fill actuals)")
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
    st.title("StockWise (Simple)")
    st.caption("Hybrid Machine Learning Stock Price Prediction (clean UI)")

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
        st.subheader("AI Prediction")
        trigger = run_prediction or st.button("Generate AI Prediction")
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

                            # Save to history
                            target_time = datetime.now() + timedelta(hours=1)
                            prediction_record = {
                                "timestamp": datetime.now(),
                                "ticker": selected_ticker,
                                "current_price": current_price,
                                "predicted_price": predicted_price,
                                "target_time": target_time,
                                "actual_price": None,
                                "error_pct": None,
                                "error_abs": None,
                                "confidence": confidence,
                                "signal": "HOLD",
                            }
                            ensure_csv_has_header(PREDICTION_HISTORY_CSV, show_cols)
                            df_hist = pd.read_csv(PREDICTION_HISTORY_CSV)
                            df_hist = pd.concat([df_hist, pd.DataFrame([prediction_record])], ignore_index=True)
                            df_hist.to_csv(PREDICTION_HISTORY_CSV, index=False)
                            st.success("Prediction saved to history.")
                        else:
                            st.error("Failed to generate prediction.")
                    else:
                        st.warning("Insufficient data for prediction (need at least 128 points).")
                except Exception as e:
                    st.error(f"Error generating prediction: {str(e)}")

        # Watchlist
        if run_watchlist:
            st.subheader("Watchlist Analysis")
            watchlist_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
            progress = st.progress(0)
            status = st.empty()
            results = []
            for i, stock in enumerate(watchlist_stocks):
                status.text(f"Analyzing {stock}...")
                try:
                    stock_data = fetch_stock_data(stock, period=f"{days_history}d", interval="60m")
                    if stock_data is not None and len(stock_data) >= SEQ_LEN:
                        latest = stock_data.iloc[-1]
                        cur_px = float(latest["Close"]) if not pd.isna(latest["Close"]) else np.nan
                        cross_d = fetch_cross_assets(stock_data.index, days_history, stock)
                        feats = create_features_for_app(stock, stock_data, cross_d)
                        if feats is not None and len(feats) >= SEQ_LEN:
                            pred_px, change_pct, votes_w, conf = predict_with_models(
                                feats, cur_px, scaler, cnn_model, lstm_model, xgb_model, meta_model
                            )
                            if pred_px is not None:
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
                except Exception as e:
                    st.warning(f"Could not analyze {stock}: {str(e)}")
                progress.progress((i + 1) / len(watchlist_stocks))
                time.sleep(0.3)

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

                colA, colB = st.columns(2)
                with colA:
                    if st.button("Refresh Actual Prices"):
                        with st.spinner("Updating actual prices..."):
                            _ = batch_evaluate_predictions_csv(PREDICTION_HISTORY_CSV)
                            st.success("Actual prices updated.")
                            st.rerun()
                with colB:
                    if st.button("View Detailed Table"):
                        df_disp = history_df.copy()
                        for col in ["timestamp", "target_time"]:
                            if col in df_disp.columns:
                                df_disp[col] = pd.to_datetime(df_disp[col], errors="coerce")
                        st.dataframe(df_disp, use_container_width=True)

                # Recent
                st.subheader("Recent Predictions")
                recent_df = history_df.sort_values("timestamp", ascending=False).head(10)
                st.dataframe(recent_df, use_container_width=True)
            else:
                st.info("No prediction history yet.")
        except Exception as e:
            st.error(f"Error loading prediction history: {str(e)}")
    else:
        st.info("No prediction history file found.")

with tab3:
    st.header("Theory: Models and Indicators (Brief)")
    st.write("This app uses an ensemble of LSTM, CNN, and XGBoost with a meta-learner to predict 1-hour forward returns, converted to price using the current price.")
    st.subheader("Models")
    st.markdown("- LSTM: captures long-range temporal dependencies")
    st.markdown("- CNN: learns local temporal patterns")
    st.markdown("- XGBoost: models tabular non-linear interactions")
    st.markdown("- Meta-learner: blends model outputs for stability")
    st.subheader("Indicators")
    st.markdown("- Trend: SMA, EMA, MACD")
    st.markdown("- Momentum: Returns, RSI")
    st.markdown("- Volatility: Rolling std of returns")
    st.markdown("- Volume: 1h change, 24h MA")
