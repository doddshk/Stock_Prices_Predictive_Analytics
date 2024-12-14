import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import yfinance as yf

def main():

    # Load stock data
    ticker = "AAPL"  # Example: Apple stock
    data = yf.download(ticker, start="2015-01-01", end="2023-12-31")
    data = data[['Close']]  # Use only the closing price
    print(data.head())

    # Scale data to range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Prepare data for LSTM
    sequence_length = 60  # Use the last 60 days to predict the next day
    X, y = [], []

    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # TRAINING
    # Split the data into training and testing sets
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Train the model
    model.fit(X_train, y_train, batch_size=32, epochs=10)

    # PREDICTIONS

    # Predict on test data
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)  # Reverse scaling

    # Reverse scale the test labels for comparison
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot the predictions vs actual values
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[split_index+sequence_length:], y_test_scaled, label="Actual Price")
    plt.plot(data.index[split_index+sequence_length:], predictions, label="Predicted Price")
    plt.title(f"{ticker} Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
