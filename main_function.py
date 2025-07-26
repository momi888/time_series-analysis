import pandas as pd 
from Arima import arima_analysis
Arima_forcast = arima_analysis

from Lstm_model  import lstm_analysis
LSTM_forcast = lstm_analysis

from Sarims import sarima_analysis
sarims_forcast = sarima_analysis

from prophet_model import statistical_analysis
prophet_forcast = statistical_analysis

from Seasonal import result
seasonal_decomposition = result

from Select_symbol import select_symbol
select_analysis = select_symbol

from rolling_statistics import plot_rolling_statistics
rolling_plot = plot_rolling_statistics

from bollinger_band import plot_bollinger_bands
bollinger_plot  = plot_bollinger_bands

from comulative_return import plot_cumulative_returns
comulative_analysis  = plot_cumulative_returns

from muliti_model_forcast import plot_forecast_comparison
multi_model_comparison = plot_forecast_comparison

def load_data():
    df = pd.read_csv(r"C:\Users\Admin\NIFTY50_all - Copy.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()


def main():
    df = load_data()
    
    symbol = "MUNDRAPORT"
    df_symbol = select_analysis(df, symbol)
    
    prophet_model = statistical_analysis(df_symbol, symbol)

    arima_model = arima_analysis(df_symbol, symbol)
    
    sarima_model = sarima_analysis(df_symbol, symbol)
    
    lstm_model = lstm_analysis(df_symbol, symbol)

    seasonal_model = seasonal_decomposition(df_symbol, symbol='MUNDRAPORT')

    rolling_statistics  = rolling_plot(df_symbol, symbol, window=30)

    bollinger_graph = bollinger_plot(df_symbol, symbol, window=20)

    comulitive_plot = comulative_analysis(df_symbol, symbol)


    multi_model_analysis  = multi_model_comparison(df_symbol, symbol, forecast_days=365)


    print("All analyses completed successfully!")

if __name__ == "__main__":
    main()

# main()