import matplotlib.pyplot as plt 
import pandas as pd 

def plot_rolling_statistics(df_symbol, symbol, window=30):
    """Plot rolling mean and standard deviation"""

    # Calculate rolling stats
    rolling_mean = df_symbol['Close'].rolling(window=window).mean()
    rolling_std = df_symbol['Close'].rolling(window=window).std()

    # Create a matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_symbol['Close'], label='Actual', color='blue')
    ax.plot(rolling_mean, label=f'{window}-Day Rolling Mean', color='red')
    ax.plot(rolling_std, label=f'{window}-Day Rolling Std', color='green')
    ax.set_title(f'{symbol} - Rolling Statistics')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)

    return fig  # ✅ Return figure to Streamlit





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

# def plot_rolling_statistics(df_symbol, symbol, window=30):
#     """Plot rolling mean and standard deviation"""
#     rolling_mean = df_symbol['Close'].rolling(window=window).mean()
#     rolling_std = df_symbol['Close'].rolling(window=window).std()
    
#     plt.figure(figsize=(12,6))
#     plt.plot(df_symbol['Close'], label='Actual', color='blue')
#     plt.plot(rolling_mean, label=f'{window}-Day Rolling Mean', color='red')
#     plt.plot(rolling_std, label=f'{window}-Day Rolling Std', color='green')
#     plt.title(f'{symbol} - Rolling Statistics')
#     plt.xlabel('Date')
#     plt.ylabel('Price')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     return rolling_mean , rolling_std

# plot_rolling_statistics(df_symbol, symbol, window=30)