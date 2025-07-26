from converted_notebook import data_import
df = data_import()

import matplotlib.pyplot as plt 

def select_symbol(df, symbol="MUNDRAPORT"):
    """Select and prepare data for a specific symbol"""
    df_symbol = df[df['Symbol'] == symbol].copy()
    df_symbol = df_symbol.sort_values('Date')
    
    plt.figure(figsize=(12,6))
    plt.plot(df_symbol['Date'], df_symbol['Close'])
    plt.title(f"{symbol} Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.show()
    
    return df_symbol

select_symbol(df,symbol = 'MUNDRAPORT')