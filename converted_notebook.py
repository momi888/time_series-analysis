import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error

def data_import():
    """Import and preprocess the NIFTY50 data"""
    df = pd.read_csv(r"C:\Users\Admin\NIFTY50_all - Copy.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    print("Data imported successfully")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Unique symbols: {df['Symbol'].unique()}")
    return df

# import tensorflow as tf
# print(tf.__version__)
