import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


# Set Streamlit app layout
st.set_page_config(layout="wide")
st.title("Stock Price Prediction with LSTM")

# User Input for Stock Ticker
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", value="AAPL")
st.write(f"You entered: {ticker}")

# Date Range Input
default_start = "2010-01-01"
default_end = "2025-12-01"
start_date = st.date_input("Start Date:", pd.to_datetime(default_start))
end_date = st.date_input("End Date:", pd.to_datetime(default_end))

if st.button("Fetch and Predict"):
    st.write(f"Fetching data for {ticker} from {start_date} to {end_date}...")

    # Step 2: Download Stock Data
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        st.error("No data found for the given ticker and date range. Please try again.")
    else:
        prices = data['Close']

        # Plot the closing price
        st.subheader("Stock Closing Price")
        plt.figure(figsize=(12, 6))
        plt.plot(prices, label='Closing Price')
        plt.title(f'{ticker} Stock Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(plt)

        # Step 3: Preprocess Data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(prices.values.reshape(-1, 1))

        def create_dataset(data, time_step=60):
            X, y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                y.append(data[i + time_step, 0])
            return np.array(X), np.array(y)

        time_step = 60
        X, y = create_dataset(scaled_prices, time_step)

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Step 4: Build and Train the LSTM Model
        st.write("Building and training the LSTM model...")
        model = tf.keras.Sequential([
             tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
             tf.keras.layers.LSTM(50, return_sequences=False),
             tf.keras.layers.Dense(25),
             tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=0)  # Fewer epochs for quicker training

        # Step 5: Evaluate the Model
        st.write("Evaluating the model...")
        predicted_prices = model.predict(X_test)
        predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))
        actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

        st.subheader("Predicted vs Actual Prices")
        plt.figure(figsize=(12, 6))
        plt.plot(actual_prices, label='Actual Prices')
        plt.plot(predicted_prices, label='Predicted Prices')
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(plt)

        # Step 6: Predict the Next Day
        st.write("Predicting the next day price...")
        last_60_days = scaled_prices[-60:]
        last_60_days = last_60_days.reshape((1, -1, 1))
        next_day_price = model.predict(last_60_days)
        next_day_price = scaler.inverse_transform(next_day_price)
        st.success(f"Predicted next day price for {ticker}: ${next_day_price[0][0]:.2f}")
