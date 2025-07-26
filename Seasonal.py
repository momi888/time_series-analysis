from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt 
import pandas as pd

def load_data():
    """Load and prepare the dataset"""
    df = pd.read_csv(r"C:\Users\Admin\NIFTY50_all.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

from Select_symbol import select_symbol
df_symbol = select_symbol(df, symbol='MUNDRAPORT')  
def seasonal_decomposition(df_symbol, symbol='MUNDRAPORT'):
    """Perform seasonal decomposition of time series data"""
   
    df_symbol = df_symbol.copy()
    
    df_symbol.set_index('Date', inplace=True)
    if not isinstance(df_symbol.index, pd.DatetimeIndex):
        df_symbol.index = pd.to_datetime(df_symbol.index)
    
    df_symbol['Close'] = df_symbol['Close'].fillna(method='ffill')
    
    result = seasonal_decompose(df_symbol['Close'], model='additive', period=30)
    
    plt.figure(figsize=(12, 8))
    result.plot()
    plt.suptitle(f"Trend, Seasonality, Residuals for {symbol}", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return result

result = seasonal_decomposition(df_symbol, symbol='MUNDRAPORT')