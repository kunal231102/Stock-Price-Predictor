# Stock Price Prediction with LSTM

A Streamlit-based web application to predict stock prices using a Long Short-Term Memory (LSTM) neural network.

## Features
- Input stock ticker and date range to fetch historical data.
- Visualize historical closing prices with interactive charts.
- Predict next-day stock price using an LSTM model.
- Compare actual vs. predicted stock prices with performance graphs.

## Technology Stack
- **Frontend:** Streamlit
- **Backend:** TensorFlow/Keras for LSTM model
- **APIs:** Yahoo Finance (`yfinance`)
- **Libraries:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`

## How It Works
1. Fetch historical stock data using Yahoo Finance.
2. Preprocess data with MinMaxScaler and create time-series datasets.
3. Train a two-layer LSTM model for stock price prediction.
4. Predict future stock prices and visualize results.

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
