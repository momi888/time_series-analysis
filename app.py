# import tensorflow as tf
# from tensorflow.keras import Dense, LSTM
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense,LSTM

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from prophet import Prophet
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.seasonal import seasonal_decompose
# from datetime import timedelta

# FORECAST_DAYS = 30

# @st.cache_data
# def load_data():
#     df = pd.read_csv(r"C:\Users\Admin\NIFTY50_all.csv")
#     df['Date'] = pd.to_datetime(df['Date'])
#     return df

# def prepare_lstm_data(df_symbol):
#     df_symbol = df_symbol.sort_values('Date')
#     close = df_symbol['Close'].values.reshape(-1, 1)
#     scaler = MinMaxScaler()
#     scaled = scaler.fit_transform(close)

#     X, y = [], []
#     for i in range(60, len(scaled)):
#         X.append(scaled[i-60:i])
#         y.append(scaled[i])

#     return np.array(X), np.array(y), scaler, df_symbol

# from tensorflow.keras import Sequential

# def train_lstm(X, y):
#     model = Sequential([
#         LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
#         LSTM(50),
#         Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mse')
#     model.fit(X, y, epochs=5, batch_size=32, verbose=0)
#     return model

# def forecast_lstm(df_symbol):
#     X, y, scaler, df_symbol = prepare_lstm_data(df_symbol)
#     model = train_lstm(X, y)
#     last_60 = X[-1].reshape(1, 60, 1)
#     forecast_scaled = []

#     for _ in range(FORECAST_DAYS):
#         pred = model.predict(last_60)[0][0]
#         forecast_scaled.append(pred)
#         last_60 = np.append(last_60[:, 1:, :], [[[pred]]], axis=1)

#     forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1))
#     last_date = df_symbol['Date'].max()
#     dates = [last_date + timedelta(days=i+1) for i in range(FORECAST_DAYS)]
#     return pd.DataFrame({'Date': dates, 'Forecast': forecast.flatten(), 'Model': 'LSTM'})

# def forecast_arima(df_symbol):
#     df_symbol = df_symbol.sort_values('Date')
#     model = ARIMA(df_symbol['Close'], order=(5,1,0)).fit()
#     forecast = model.forecast(steps=FORECAST_DAYS)
#     dates = [df_symbol['Date'].max() + timedelta(days=i+1) for i in range(FORECAST_DAYS)]
#     return pd.DataFrame({'Date': dates, 'Forecast': forecast, 'Model': 'ARIMA'})

# def forecast_sarima(df_symbol):
#     df_symbol = df_symbol.sort_values('Date')
#     model = SARIMAX(df_symbol['Close'], order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
#     forecast = model.forecast(steps=FORECAST_DAYS)
#     dates = [df_symbol['Date'].max() + timedelta(days=i+1) for i in range(FORECAST_DAYS)]
#     return pd.DataFrame({'Date': dates, 'Forecast': forecast, 'Model': 'SARIMA'})

# def forecast_prophet(df_symbol):
#     prophet_df = df_symbol[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
#     model = Prophet()
#     model.fit(prophet_df)
#     future = model.make_future_dataframe(periods=FORECAST_DAYS)
#     forecast = model.predict(future)
#     result = forecast[['ds', 'yhat']].tail(FORECAST_DAYS).rename(columns={'ds': 'Date', 'yhat': 'Forecast'})
#     result['Model'] = 'Prophet'
#     return result

# def show_decomposition(df_symbol):
#     df_symbol = df_symbol.sort_values('Date')
#     result = seasonal_decompose(df_symbol['Close'], model='additive', period=30)
#     fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
#     result.observed.plot(ax=axs[0], title='Observed')
#     result.trend.plot(ax=axs[1], title='Trend')
#     result.seasonal.plot(ax=axs[2], title='Seasonal')
#     result.resid.plot(ax=axs[3], title='Residual')
#     plt.tight_layout()
#     return fig

# st.title(" NIFTY50 Stock Forecast Dashboard (30 Days)")
# df = load_data()
# symbols = sorted(df['Symbol'].unique())
# symbol = st.selectbox("Select Stock Symbol", symbols)

# if st.button("Run Forecast"):
#     df_symbol = df[df['Symbol'] == symbol]

#     st.write("Running forecasts for next 30 days...")

#     df_lstm = forecast_lstm(df_symbol)
#     df_arima = forecast_arima(df_symbol)
#     df_sarima = forecast_sarima(df_symbol)
#     df_prophet = forecast_prophet(df_symbol)

#     combined = pd.concat([df_lstm, df_arima, df_sarima, df_prophet])

#     st.subheader("Forecast Comparison (30 Days)")
#     fig, ax = plt.subplots()
#     for model in combined['Model'].unique():
#         subset = combined[combined['Model'] == model]
#         ax.plot(subset['Date'], subset['Forecast'], label=model)

#     ax.set_title(f"{symbol} – Forecast (LSTM, ARIMA, SARIMA, Prophet)")
#     ax.legend()
#     st.pyplot(fig)

#     st.subheader("Forecast Table")
#     st.dataframe(combined)

#     csv = combined.to_csv(index=False).encode('utf-8')
#     st.download_button("Download Forecast CSV", csv, file_name=f"{symbol}_30day_forecast.csv", mime='text/csv')
    
#     with st.expander("Show Seasonal Decomposition (Last 30 Days)"):
#         fig2 = show_decomposition(df_symbol)
#         st.pyplot(fig2)



# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense, LSTM
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM
# import tensorflow as tf
# from tensorflow import keras           # Should work
# import streamlit as st
# import pandas as pd
# import numpy as np
# # import tensorflow.compat.v2 as tf 
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from prophet import Prophet
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.seasonal import seasonal_decompose
# from datetime import timedelta

# FORECAST_DAYS = 30

# @st.cache_data
# def load_data():
#     df = pd.read_csv(r"C:\Users\Admin\NIFTY50_all.csv")
#     df['Date'] = pd.to_datetime(df['Date'])
#     return df

# def prepare_lstm_data(df_symbol):
#     df_symbol = df_symbol.sort_values('Date')
#     close = df_symbol['Close'].values.reshape(-1, 1)
#     scaler = MinMaxScaler()
#     scaled = scaler.fit_transform(close)

#     X, y = [], []
#     for i in range(60, len(scaled)):
#         X.append(scaled[i-60:i])
#         y.append(scaled[i])

#     return np.array(X), np.array(y), scaler, df_symbol

# def train_lstm(X, y):
#     model = Sequential([
#         LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
#         LSTM(50),
#         Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mse')
#     model.fit(X, y, epochs=5, batch_size=32, verbose=0)
#     return model

# def forecast_lstm(df_symbol):
#     X, y, scaler, df_symbol = prepare_lstm_data(df_symbol)
#     model = train_lstm(X, y)
#     last_60 = X[-1].reshape(1, 60, 1)
#     forecast_scaled = []

#     for _ in range(FORECAST_DAYS):
#         pred = model.predict(last_60, verbose=0)[0][0]
#         forecast_scaled.append(pred)
#         last_60 = np.append(last_60[:, 1:, :], [[[pred]]], axis=1)

#     forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1))
#     last_date = df_symbol['Date'].max()
#     dates = [last_date + timedelta(days=i+1) for i in range(FORECAST_DAYS)]
#     return pd.DataFrame({'Date': dates, 'Forecast': forecast.flatten(), 'Model': 'LSTM'})

# def forecast_arima(df_symbol):
#     df_symbol = df_symbol.sort_values('Date')
#     model = ARIMA(df_symbol['Close'], order=(5,1,0)).fit()
#     forecast = model.forecast(steps=FORECAST_DAYS)
#     dates = [df_symbol['Date'].max() + timedelta(days=i+1) for i in range(FORECAST_DAYS)]
#     return pd.DataFrame({'Date': dates, 'Forecast': forecast, 'Model': 'ARIMA'})

# def forecast_sarima(df_symbol):
#     df_symbol = df_symbol.sort_values('Date')
#     model = SARIMAX(df_symbol['Close'], order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
#     forecast = model.forecast(steps=FORECAST_DAYS)
#     dates = [df_symbol['Date'].max() + timedelta(days=i+1) for i in range(FORECAST_DAYS)]
#     return pd.DataFrame({'Date': dates, 'Forecast': forecast, 'Model': 'SARIMA'})

# def forecast_prophet(df_symbol):
#     prophet_df = df_symbol[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
#     model = Prophet()
#     model.fit(prophet_df)
#     future = model.make_future_dataframe(periods=FORECAST_DAYS)
#     forecast = model.predict(future)
#     result = forecast[['ds', 'yhat']].tail(FORECAST_DAYS).rename(columns={'ds': 'Date', 'yhat': 'Forecast'})
#     result['Model'] = 'Prophet'
#     return result

# def show_decomposition(df_symbol):
#     df_symbol = df_symbol.sort_values('Date')
#     result = seasonal_decompose(df_symbol['Close'], model='additive', period=30)
#     fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
#     result.observed.plot(ax=axs[0], title='Observed')
#     result.trend.plot(ax=axs[1], title='Trend')
#     result.seasonal.plot(ax=axs[2], title='Seasonal')
#     result.resid.plot(ax=axs[3], title='Residual')
#     plt.tight_layout()
#     return fig

# def main():
#     st.title("NIFTY50 Stock Forecast Dashboard (30 Days)")
#     df = load_data()
#     symbols = sorted(df['Symbol'].unique())
#     symbol = st.selectbox("Select Stock Symbol", symbols)

#     if st.button("Run Forecast"):
#         df_symbol = df[df['Symbol'] == symbol]

#         st.write("Running forecasts for next 30 days...")

#         df_lstm = forecast_lstm(df_symbol)
#         df_arima = forecast_arima(df_symbol)
#         df_sarima = forecast_sarima(df_symbol)
#         df_prophet = forecast_prophet(df_symbol)

#         combined = pd.concat([df_lstm, df_arima, df_sarima, df_prophet])

#         st.subheader("Forecast Comparison (30 Days)")
#         fig, ax = plt.subplots()
#         for model in combined['Model'].unique():
#             subset = combined[combined['Model'] == model]
#             ax.plot(subset['Date'], subset['Forecast'], label=model)

#         ax.set_title(f"{symbol} - Forecast (LSTM, ARIMA, SARIMA, Prophet)")
#         ax.legend()
#         st.pyplot(fig)

#         st.subheader("Forecast Table")
#         st.dataframe(combined)

#         csv = combined.to_csv(index=False).encode('utf-8')
#         st.download_button(
#             "Download Forecast CSV",
#             csv,
#             file_name=f"{symbol}_30day_forecast.csv",
#             mime='text/csv'
#         )
        
#         with st.expander("Show Seasonal Decomposition (Last 30 Days)"):
#             fig2 = show_decomposition(df_symbol)
#             st.pyplot(fig2)

# if __name__ == "__main__":
#     main()