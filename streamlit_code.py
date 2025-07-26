import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import timedelta


st.set_page_config(
    page_title="NIFTY50 Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stSelectbox>div>div>select {
        background-color: #ffffff;
    }
    .css-1v3fvcr {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .plot-container {
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("NIFTY50 Stock Analysis Dashboard")
st.markdown("""
Comprehensive analysis of NIFTY50 stocks with multiple forecasting models and technical indicators.
""")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r"C:\Users\Admin\NIFTY50_all - Copy.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        
        df.columns = df.columns.str.strip() 
        df.columns = df.columns.str.replace(' ', '_') 
        
        volume_aliases = ['Total_Traded_Quantity', 'Traded_Quantity', 'Volume', 'QTY', 'Total_Trade_Quantity']
        for alias in volume_aliases:
            if alias in df.columns:
                df.rename(columns={alias: 'Volume'}, inplace=True)
                break
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame() 

@st.cache_data
def get_symbol_data(df, symbol):
    try:
        df_symbol = df[df['Symbol'] == symbol].copy()
        df_symbol = df_symbol.sort_values('Date')
        return df_symbol
    except Exception as e:
        st.error(f"Error processing symbol data: {str(e)}")
        return pd.DataFrame()

def run_analysis(df, symbol):
    if df.empty:
        st.warning("No data available for analysis")
        return
    
    df_symbol = get_symbol_data(df, symbol)
    
    if df_symbol.empty:
        st.warning(f"No data available for symbol: {symbol}")
        return
    
    st.header("📈 Basic Analysis")

    st.subheader("1. Closing Price History")
    try:
        fig1, ax1 = plt.subplots(figsize=(10,6))
        ax1.plot(df_symbol['Date'], df_symbol['Close'])
        ax1.set_title(f"{symbol} Closing Price")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price")
        st.pyplot(fig1)
        plt.close(fig1)
    except Exception as e:
        st.error(f"Error plotting closing price: {str(e)}")
    
    st.subheader("2. Volume Analysis")
    try:
        if 'Volume' in df_symbol.columns:
            fig2, ax2 = plt.subplots(figsize=(10,6))
            ax2.plot(df_symbol['Date'], df_symbol['Volume'])
            ax2.set_title(f"{symbol} Trading Volume")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Volume")
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.warning("Volume data not available in the dataset")
    except Exception as e:
        st.error(f"Error plotting volume: {str(e)}")
    
    st.subheader("3. Seasonal Decomposition")
    try:
        from Seasonal import seasonal_decomposition
        result = seasonal_decomposition(df_symbol, symbol)
        st.pyplot(plt.gcf())
        plt.close()
    except Exception as e:
        st.error(f"Error in seasonal decomposition: {str(e)}")
    
    st.header("🔮 Forecasting Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("4. ARIMA Forecast")
        try:
            from Arima import arima_analysis
            df_arima = df_symbol.copy().set_index('Date')
            df_arima = df_arima.asfreq('B').fillna(method='ffill')
            fig4 = arima_analysis(df_arima, symbol)
            st.pyplot(fig4)
            plt.close(fig4)
        except Exception as e:
            st.error(f"ARIMA Error: {str(e)}")
        
        st.subheader("5. SARIMA Forecast")
        try:
            from Sarims import sarima_analysis

    # Properly copy and set index
            df_sarima = df_symbol.copy(deep=True)
            df_sarima.set_index('Date', inplace=True)

    # Filter data before 2011
            df_sarima = df_sarima[df_sarima.index < "2011-01-01"]

    # Generate SARIMA forecast plot
            fig5 = sarima_analysis(df_sarima, symbol)
            st.pyplot(fig5)
            plt.close(fig5)

        except Exception as e:
            st.error(f"SARIMA Error: {str(e)}")

        

    
    with col2:
        st.subheader("6. Prophet Forecast")
        try:
            from prophet_model import statistical_analysis
            fig6 = statistical_analysis(df_symbol, symbol)
            st.pyplot(fig6)
            plt.close(fig6)
        except Exception as e:
            st.error(f"Prophet Error: {str(e)}")
        
    with col2:
        st.subheader("7. LSTM Forecast")
        try:
            from Lstm_model import lstm_analysis
            fig7 = lstm_analysis(df_symbol, symbol)
            st.pyplot(fig7)
            plt.close()  # Or use plt.close(fig7) if you’re managing multiple figs
        except Exception as e:
            st.error(f"LSTM Error: {str(e)}")

    
    st.header("📊 Technical Indicators")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("8. Rolling Statistics")
        try:
            from rolling_statistics import plot_rolling_statistics
            fig8 = plot_rolling_statistics(df_symbol, symbol)
            st.pyplot(fig8)
            plt.close(fig8)
        except Exception as e:
            st.error(f"Rolling Stats Error: {str(e)}")
    
    with col4:
        st.subheader("9. Bollinger Bands")
        try:
            from bollinger_band import plot_bollinger_bands
            fig9 = plot_bollinger_bands(df_symbol, symbol)
            st.pyplot(fig9)
            plt.close(fig9)
        except Exception as e:
            st.error(f"Bollinger Bands Error: {str(e)}")

    
    st.subheader("10. Cumulative Returns")
    try:
        from comulative_return import plot_cumulative_returns
        fig10 = plot_cumulative_returns(df_symbol, symbol)
        st.pyplot(fig10)
        plt.close(fig10)
    except Exception as e:
        st.error(f"Cumulative Returns Error: {str(e)}")

def main():
    try:
        df = load_data()
        
        if df.empty:
            st.error("Failed to load data. Please check the data file.")
            return
            
        if 'Symbol' not in df.columns:
            st.error("The dataset doesn't contain a 'Symbol' column.")
            return
            
        symbol = st.selectbox("Select Stock Symbol", df['Symbol'].unique(), index=0)
        run_analysis(df, symbol)
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()