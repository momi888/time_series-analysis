import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def arima_analysis(df_symbol, symbol):
    """Perform ARIMA forecasting"""

    # Ensure index is datetime
    if not isinstance(df_symbol.index, pd.DatetimeIndex):
        df_symbol.index = pd.to_datetime(df_symbol.index)

    # Set frequency to business days and fill missing values
    df_symbol = df_symbol.asfreq('B')
    df_symbol = df_symbol.fillna(method='ffill')

    # Fit ARIMA model
    model = ARIMA(df_symbol['Close'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=365)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(df_symbol['Close'], label='Observed')
    forecast_index = pd.date_range(df_symbol.index[-1], periods=366, freq='B')[1:]
    plt.plot(forecast_index, forecast, label='ARIMA Forecast')
    plt.legend()
    plt.title(f"{symbol} Close Price Forecast with ARIMA")
    plt.grid()
    plt.tight_layout()

    return plt  # Return the plot object for Streamlit



# import matplotlib.pyplot as plt
# import pandas as pd
# from statsmodels.tsa.arima.model import ARIMA
# from Select_symbol import select_symbol

# # Correct: actually call the function
# df_symbol = select_symbol()
# symbol = 'MUNDRAPORT'

# def arima_analysis(df_symbol, symbol):
#     """Perform ARIMA forecasting"""
    
#     # Ensure index is datetime
#     if not isinstance(df_symbol.index, pd.DatetimeIndex):
#         df_symbol.index = pd.to_datetime(df_symbol.index)

#     df_symbol = df_symbol.asfreq('B')  # Business day frequency
#     df_symbol = df_symbol.fillna(method='ffill')  # Fill missing values

#     model = ARIMA(df_symbol['Close'], order=(5, 1, 0))
#     model_fit = model.fit()
#     forecast = model_fit.forecast(steps=365)

#     plt.figure(figsize=(12, 6))
#     plt.plot(df_symbol['Close'], label='Observed')
#     plt.plot(pd.date_range(df_symbol.index[-1], periods=365, freq='B')[1:], forecast, label='ARIMA Forecast')
#     plt.legend()
#     plt.title(f"{symbol} Close Price Forecast with ARIMA")
#     plt.grid()
#     plt.tight_layout()
#     plt.show()

#     return plt  # return figure for Streamlit


# import matplotlib.pyplot as plt
# from statsmodels.tsa.arima.model import ARIMA 
# from Select_symbol import select_symbol
# df_symbol = select_symbol
# symbol = 'MUNDRAPORT'

# def arima_analysis(df_symbol, symbol):
#     """Perform ARIMA forecasting"""
#     df_symbol = df_symbol.asfreq('B') 
#     df_symbol = df_symbol.fillna(method='ffill')
    
#     model = ARIMA(df_symbol['Close'], order=(5,1,0))
#     model_fit = model.fit()
#     forecast = model_fit.forecast(steps=365)
    
#     plt.figure(figsize=(12,6))
#     plt.plot(df_symbol['Close'], label='Observed')
#     plt.plot(pd.date_range(df_symbol.index[-1], periods=365, freq='B'), forecast, label='ARIMA Forecast')
#     plt.legend()
#     plt.title(f"{symbol} Close Price Forecast with ARIMA")
#     plt.show()
    
#     return model_fit

# arima_analysis(df_symbol, symbol)