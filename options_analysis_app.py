import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# Function to fetch stock data
def get_stock_data(symbol, interval='1d', days=90):
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=days)
    stock = yf.Ticker(symbol)
    return stock.history(interval=interval, start=start_date, end=end_date)

# Function to calculate indicators
def calculate_indicators(data):
    data['MA10'] = data['Close'].rolling(10, min_periods=1).mean()
    data['MA50'] = data['Close'].rolling(50, min_periods=1).mean()
    delta = data['Close'].diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    return data

# Function to detect support and resistance
def detect_support_resistance(data):
    local_max = argrelextrema(data['Close'].values, np.greater, order=5)[0]
    local_min = argrelextrema(data['Close'].values, np.less, order=5)[0]
    return local_max, local_min

# Function to fetch options chain
def fetch_options_chain(symbol):
    stock = yf.Ticker(symbol)
    expiry_dates = stock.options
    if not expiry_dates:
        return None, None
    options_chain = stock.option_chain(expiry_dates[0])
    calls = options_chain.calls.sort_values(by=['openInterest', 'volume'], ascending=False).head(3)
    puts = options_chain.puts.sort_values(by=['openInterest', 'volume'], ascending=False).head(3)
    return calls, puts

# Streamlit App
st.title("Options Analysis and Technical Indicators")
ticker = st.text_input("Enter Stock Ticker (e.g., TSLA, AAPL):", "")

if ticker:
    st.write(f"### Analyzing {ticker.upper()}...")
    data = get_stock_data(ticker, days=90)
    
    if not data.empty:
        # Calculate indicators
        data = calculate_indicators(data)
        local_max, local_min = detect_support_resistance(data)
        calls, puts = fetch_options_chain(ticker)

        # Display stock chart with support and resistance
        st.write("#### Stock Chart with Support and Resistance")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data.index, data['Close'], label='Close Price', color='black')
        ax.plot(data['MA10'], label='MA10', linestyle='--', color='blue')
        ax.plot(data['MA50'], label='MA50', linestyle='--', color='red')
        ax.scatter(data.index[local_max], data['Close'].iloc[local_max], color='red', label='Resistance', marker='^')
        ax.scatter(data.index[local_min], data['Close'].iloc[local_min], color='green', label='Support', marker='v')
        ax.legend()
        st.pyplot(fig)

        # Display key indicators
        st.write("#### Key Indicators")
        st.write(f"**Latest Close Price:** ${data['Close'].iloc[-1]:.2f}")
        st.write(f"**RSI:** {data['RSI'].iloc[-1]:.2f}")
        st.write(f"**VWAP:** ${data['VWAP'].iloc[-1]:.2f}")

        # Display options chain
        if calls is not None and puts is not None:
            st.write("#### Top Call Options")
            st.dataframe(calls[['strike', 'lastPrice', 'openInterest', 'volume']])
            st.write("#### Top Put Options")
            st.dataframe(puts[['strike', 'lastPrice', 'openInterest', 'volume']])
        else:
            st.write("No options data available for this stock.")
    else:
        st.write("No data available for this ticker. Please check the symbol.")
