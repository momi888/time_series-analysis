import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

def statistical_analysis(df_symbol, symbol):
    """Forecast using Prophet"""

    df = df_symbol.copy()

    # Prophet requires 'ds' and 'y' columns
    df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    fig = model.plot(forecast)
    plt.title(f"{symbol} Forecast using Prophet")
    plt.tight_layout()

    return plt  # Return the figure object to Streamlit



