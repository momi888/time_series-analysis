import matplotlib.pyplot as plt 
import pandas as pd 

def plot_bollinger_bands(df_symbol, symbol, window=20):
    """Plot Bollinger Bands for volatility analysis"""

    rolling_mean = df_symbol['Close'].rolling(window).mean()
    rolling_std = df_symbol['Close'].rolling(window).std()
    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)

    # Create a Matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_symbol['Close'], label='Close Price', color='blue')
    ax.plot(rolling_mean, label='Rolling Mean', color='black')
    ax.plot(upper_band, label='Upper Band', color='red', linestyle='--')
    ax.plot(lower_band, label='Lower Band', color='green', linestyle='--')
    ax.fill_between(df_symbol.index, upper_band, lower_band, color='grey', alpha=0.1)
    ax.set_title(f'{symbol} - Bollinger Bands (Window={window})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)

    return fig  # ✅ Return figure to be used with st.pyplot



# import matplotlib.pyplot as plt 
# import pandas as pd 

# from Select_symbol import select_symbol

# def load_data():
#     """Load and prepare the dataset"""
#     df = pd.read_csv(r"C:\Users\Admin\NIFTY50_all - Copy.csv")
#     df['Date'] = pd.to_datetime(df['Date'])
#     return df


# df = load_data()
# symbol = 'MUNDRAPORT'
# df_symbol = select_symbol(df, symbol) 


# def plot_bollinger_bands(df_symbol, symbol, window=20):
#     """Plot Bollinger Bands for volatility analysis"""
#     rolling_mean = df_symbol['Close'].rolling(window).mean()
#     rolling_std = df_symbol['Close'].rolling(window).std()
#     upper_band = rolling_mean + (2 * rolling_std)
#     lower_band = rolling_mean - (2 * rolling_std)
    
#     plt.figure(figsize=(12,6))
#     plt.plot(df_symbol['Close'], label='Close Price', color='blue')
#     plt.plot(rolling_mean, label='Rolling Mean', color='black')
#     plt.plot(upper_band, label='Upper Band', color='red', linestyle='--')
#     plt.plot(lower_band, label='Lower Band', color='green', linestyle='--')
#     plt.fill_between(df_symbol.index, upper_band, lower_band, color='grey', alpha=0.1)
#     plt.title(f'{symbol} - Bollinger Bands (Window={window})')
#     plt.xlabel('Date')
#     plt.ylabel('Price')
#     plt.legend()
#     plt.grid(True)
#     plt.show()


#     return rolling_mean,rolling_std,upper_band,lower_band

# plot_bollinger_bands(df_symbol, symbol, window=20)