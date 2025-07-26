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

def plot_cumulative_returns(df_symbol, symbol):
    """Plot cumulative returns over time"""
    df_symbol['Daily Return'] = df_symbol['Close'].pct_change()
    df_symbol['Cumulative Return'] = (1 + df_symbol['Daily Return']).cumprod()
    
    plt.figure(figsize=(12,6))
    plt.plot(df_symbol['Cumulative Return'], color='purple')
    plt.title(f'{symbol} - Cumulative Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.grid(True)
    plt.show()

plot_cumulative_returns(df_symbol, symbol)