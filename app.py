"""
Stock Price Forecasting App
Downloads stock data with yfinance and forecasts future prices with Prophet.

Run locally:  streamlit run app.py
"""

from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from prophet import Prophet

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Stock Price Forecasting", page_icon="📈", layout="wide")

STOCKS = {
    "AAPL": "Apple (AAPL)",
    "GOOGL": "Google (GOOGL)",
    "MSFT": "Microsoft (MSFT)",
    "TSLA": "Tesla (TSLA)",
    "AMZN": "Amazon (AMZN)",
    "META": "Meta (META)",
    "NFLX": "Netflix (NFLX)",
    "NVDA": "NVIDIA (NVDA)",
}

MIN_ROWS = 60  # Prophet needs a reasonable amount of history


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Download daily data and return a DataFrame with a single 'Price' column."""
    data = yf.download(
        ticker,
        start=start,
        end=end + timedelta(days=1),  # yfinance's end date is exclusive
        progress=False,
        auto_adjust=True,
    )
    if data is None or data.empty:
        return pd.DataFrame()

    # Newer yfinance versions return MultiIndex columns like ('Close', 'AAPL')
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    df = data[["Close"]].copy()
    df.columns = ["Price"]
    df.index = pd.to_datetime(df.index).tz_localize(None)  # Prophet needs tz-naive dates
    df.index.name = "Date"
    return df.dropna()


@st.cache_data(show_spinner=False)
def run_forecast(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Fit Prophet on the closing prices and forecast `horizon` business days."""
    df_prophet = df["Price"].reset_index()
    df_prophet.columns = ["ds", "y"]

    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
    )
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=horizon, freq="B")
    return model.predict(future)


# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")

ticker = st.sidebar.selectbox(
    "Select Stock",
    options=list(STOCKS.keys()),
    format_func=lambda x: STOCKS[x],
)

today = date.today()
start_date = st.sidebar.date_input(
    "Start Date", value=today - timedelta(days=5 * 365), max_value=today
)
end_date = st.sidebar.date_input("End Date", value=today, max_value=today)

horizon = st.sidebar.slider("Forecast Horizon (days)", 7, 120, 90)

show_rolling = st.sidebar.checkbox("Show rolling mean", value=True)
rolling_window = st.sidebar.slider("Rolling window (days)", 5, 200, 50, disabled=not show_rolling)

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("📈 Stock Price Forecasting App")
st.caption("Historical data from Yahoo Finance (yfinance) · Forecasts by Prophet")

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

with st.spinner(f"Downloading {ticker} data..."):
    try:
        df = load_data(ticker, start_date, end_date)
    except Exception as e:
        st.error(f"Could not download data: {e}")
        st.stop()

if df.empty:
    st.error("No data returned. Try a different date range or try again later.")
    st.stop()

if len(df) < MIN_ROWS:
    st.warning(
        f"Only {len(df)} trading days found. Please choose a longer range "
        f"(at least {MIN_ROWS} trading days) for a meaningful forecast."
    )
    st.stop()

# --- Key metrics ------------------------------------------------------------
st.subheader(f"{STOCKS[ticker]} — Key Metrics")

current_price = df["Price"].iloc[-1]
prev_price = df["Price"].iloc[-2]
change_pct = (current_price - prev_price) / prev_price * 100
last_year = df[df.index >= df.index[-1] - pd.Timedelta(days=365)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Price", f"${current_price:,.2f}", f"{change_pct:+.2f}%")
c2.metric("52-Week High", f"${last_year['Price'].max():,.2f}")
c3.metric("52-Week Low", f"${last_year['Price'].min():,.2f}")
c4.metric("Trading Days Loaded", f"{len(df):,}")

# --- Historical chart -------------------------------------------------------
st.subheader("Historical Closing Prices")

hist_chart = df.rename(columns={"Price": "Close"})
if show_rolling:
    hist_chart[f"{rolling_window}-Day Rolling Mean"] = (
        hist_chart["Close"].rolling(rolling_window).mean()
    )
st.line_chart(hist_chart, y_label="Price (USD)")

with st.expander("View raw data"):
    st.dataframe(df.sort_index(ascending=False).style.format({"Price": "${:,.2f}"}))

# --- Forecast ---------------------------------------------------------------
st.subheader(f"Prophet Forecast — Next {horizon} Business Days")

with st.spinner("Fitting Prophet model..."):
    forecast = run_forecast(df, horizon)

last_hist_date = df.index[-1]
future_part = forecast[forecast["ds"] > last_hist_date]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df.index, df["Price"], label="Historical Close", color="#1f77b4", linewidth=1.2)
ax.plot(forecast["ds"], forecast["yhat"], label="Prophet Fit / Forecast",
        color="#ff7f0e", linewidth=1.2)
ax.fill_between(
    future_part["ds"], future_part["yhat_lower"], future_part["yhat_upper"],
    color="#ff7f0e", alpha=0.2, label="Uncertainty Interval",
)
ax.axvline(last_hist_date, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.set_title(f"{ticker} Price Forecast")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close(fig)

# Forecast summary
final = future_part.iloc[-1]
expected_change = (final["yhat"] - current_price) / current_price * 100
f1, f2, f3 = st.columns(3)
f1.metric(f"Forecast on {final['ds']:%Y-%m-%d}", f"${final['yhat']:,.2f}", f"{expected_change:+.2f}%")
f2.metric("Lower Bound", f"${final['yhat_lower']:,.2f}")
f3.metric("Upper Bound", f"${final['yhat_upper']:,.2f}")

# Forecast table + CSV download
forecast_table = future_part[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
forecast_table.columns = ["Date", "Forecast", "Lower Bound", "Upper Bound"]
forecast_table["Date"] = forecast_table["Date"].dt.date
forecast_table[["Forecast", "Lower Bound", "Upper Bound"]] = (
    forecast_table[["Forecast", "Lower Bound", "Upper Bound"]].round(2)
)

with st.expander("View forecast table"):
    st.dataframe(forecast_table, hide_index=True)

st.download_button(
    "⬇️ Download Forecast (CSV)",
    data=forecast_table.to_csv(index=False).encode("utf-8"),
    file_name=f"{ticker}_forecast_{horizon}d.csv",
    mime="text/csv",
)

st.info(
    "⚠️ This forecast is for educational purposes only. Stock prices are influenced by "
    "many factors a time-series model cannot capture, so do not use it for investment decisions."
)
