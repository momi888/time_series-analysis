from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

def lstm_analysis(df_symbol, symbol):
    """Perform LSTM forecasting and return the forecast plot figure"""

    # Reset index if needed
    if df_symbol.index.name == 'Date':
        df_symbol = df_symbol.reset_index()

    close_prices = df_symbol['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(close_prices)

    dfv = df_symbol[['Date', 'Close']].set_index('Date').sort_index()
    scaled = scaler.fit_transform(dfv)

    sequence_length = 60
    X, y = [], []
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i-sequence_length:i, 0])
        y.append(scaled[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    train_size = int(len(X)*0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build and train LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Predict and inverse transform
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_predict_inv = scaler.inverse_transform(train_predict.reshape(-1, 1))
    test_predict_inv = scaler.inverse_transform(test_predict.reshape(-1, 1))
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Create DataFrame for plotting
    plot_data = dfv.copy()
    plot_data['Actual'] = scaler.inverse_transform(scaled)
    plot_data['Train Predict'] = np.nan
    plot_data['Test Predict'] = np.nan
    plot_data.iloc[sequence_length:train_size+sequence_length, plot_data.columns.get_loc('Train Predict')] = train_predict_inv.flatten()
    plot_data.iloc[train_size+sequence_length:, plot_data.columns.get_loc('Test Predict')] = test_predict_inv.flatten()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(plot_data.index, plot_data['Actual'], label="Actual", color='blue')
    ax.plot(plot_data.index, plot_data['Train Predict'], label="Train Predict", color='orange')
    ax.plot(plot_data.index, plot_data['Test Predict'], label="Test Predict", color='green')
    ax.set_title(f"LSTM Forecast - {symbol}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    ax.legend()
    ax.grid(True)

    return fig  # ✅ Streamlit compatible



