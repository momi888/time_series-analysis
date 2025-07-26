# import pandas as pd
# from prophet import Prophet
# import matplotlib.pyplot as plt

# def statistical_analysis(df_symbol, symbol):
#     """Forecast using Prophet"""

#     df = df_symbol.copy()

#     # Prophet requires columns named 'ds' and 'y'
#     df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
#     df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

#     model = Prophet(daily_seasonality=True)
#     model.fit(df_prophet)

#     future = model.make_future_dataframe(periods=365)
#     forecast = model.predict(future)

#     fig = model.plot(forecast)
#     plt.title(f"{symbol} Forecast using Prophet")
#     plt.tight_layout()

#     return plt  # return for Streamlit


import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarima_analysis(df_symbol, symbol):
    """Perform SARIMA forecasting"""

    # Ensure index is datetime
    if not isinstance(df_symbol.index, pd.DatetimeIndex):
        df_symbol.index = pd.to_datetime(df_symbol.index)

    # Limit to data before 2011
    df_sarima = df_symbol[df_symbol.index < "2011-01-01"]

    # Fit SARIMA model
    model_sarima = SARIMAX(df_sarima['Close'],
                           order=(1, 1, 1),
                           seasonal_order=(1, 1, 1, 12))
    model_sarima_fit = model_sarima.fit()

    # Forecast future prices
    forecast_sarima = model_sarima_fit.forecast(steps=365)

    # Plot observed + forecast
    plt.figure(figsize=(12, 6))
    plt.plot(df_sarima['Close'], label='Observed')
    forecast_index = pd.date_range(df_sarima.index[-1], periods=366, freq='B')[1:]
    plt.plot(forecast_index, forecast_sarima, label='SARIMA Forecast')
    plt.legend()
    plt.title(f"{symbol} Close Price Forecast with SARIMA")
    plt.grid()
    plt.tight_layout()

    return plt  # Return for Streamlit
