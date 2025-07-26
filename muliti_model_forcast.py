import matplotlib.pyplot as plt 
import pandas as pd 

from Select_symbol import select_symbol

def load_data():
    """Load and prepare the dataset"""
    df = pd.read_csv(r"C:\Users\Admin\NIFTY50_all - Copy.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()
symbol = 'MUNDRAPORT'
df_symbol = select_symbol(df, symbol) 

from Arima import arima_analysis
Arima_forcast = arima_analysis

from Lstm_model  import lstm_analysis
LSTM_forcast = lstm_analysis

from Sarims import sarima_analysis
sarims_forcast = sarima_analysis

from prophet_model import statistical_analysis
prophet_forcast = statistical_analysis

def plot_forecast_comparison(df_symbol, symbol, forecast_days=365):
    """Compare forecasts from different models"""
    
    plt.figure(figsize=(14,7))
    plt.plot(df_symbol['Close'], label='Historical', color='black')
    
    plt.plot(Arima_forcast, label='ARIMA Forecast', color='red')
    plt.plot(sarims_forcast, label='SARIMA Forecast', color='blue')
    plt.plot(prophet_forcast, label='Prophet Forecast', color='green')
    plt.plot(LSTM_forcast, label='LSTM Forecast', color='purple')
    
    plt.title(f'{symbol} - Multi-Model Forecast Comparison')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_forecast_comparison(df_symbol, symbol, forecast_days=365)