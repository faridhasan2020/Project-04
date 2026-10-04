# 📈 Stock Price Forecasting App

A Streamlit web application that downloads historical stock data with **yfinance** and forecasts future closing prices with **Prophet**.

**Live App:** https://stockmarketdata.streamlit.app

## Features

- **Stock selection:** choose from Apple, Google, Microsoft, Tesla, Amazon, Meta, Netflix and NVIDIA
- **Date range selector:** pick the start and end dates of the historical data
- **Historical price chart:** interactive line chart of daily closing prices
- **Prophet forecast:** forecast 7 to 120 business days ahead with an uncertainty interval
- **Key metrics:** current price, daily change, 52-week high and 52-week low
- **Rolling mean:** optional moving-average line with an adjustable window
- **CSV export:** download the forecast as a CSV file

## How It Works

1. The closing prices of the selected stock are downloaded with `yf.download()`.
2. The data is reshaped into Prophet's format: `ds` (date) and `y` (price).
3. A Prophet model with weekly and yearly seasonality is fitted.
4. Future dates are generated with business-day frequency (`freq='B'`), so weekends are skipped.
5. The forecast and its uncertainty interval are plotted and shown in a table.

## Project Structure

```
stock-forecast-app/
├── app.py              # Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Run Locally

```bash
git clone https://github.com/faridhasan2020/Project-04/stock-forecast-app.git
cd stock-forecast-app

python -m venv venv
# Windows (PowerShell):
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

## Deployment (Streamlit Community Cloud)

1. Push `app.py`, `requirements.txt` and `README.md` to a public GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Select the repository, set the main file path to `app.py`.
5. Under **Advanced settings**, choose Python 3.11 or 3.12.
6. Choose an app URL and click **Deploy**.

## Tech Stack

Python · Streamlit · yfinance · Prophet · pandas · NumPy · Matplotlib

## Disclaimer

This project is for educational purposes only and is not financial advice.
