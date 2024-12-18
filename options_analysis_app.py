import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import datetime

# Fetch stock or ETF data with live price fallback
def get_stock_data(symbol, interval='1d', days=180):
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=days)
    stock = yf.Ticker(symbol)
    history = stock.history(interval=interval, start=start_date, end=end_date, actions=False, auto_adjust=True)
    
    # Fetch live price and pre/post-market price
    live_price = stock.info.get("currentPrice", None)
    pre_price = stock.info.get("preMarketPrice", None)
    post_price = stock.info.get("postMarketPrice", None)
    pre_post_price = pre_price or post_price or None

    # Fallback to last close price if live price is unavailable
    if live_price is None and not history.empty:
        live_price = history['Close'].iloc[-1]

    # Append live price to history
    if live_price:
        live_data = pd.DataFrame({
            'Date': [datetime.datetime.now()],
            'Close': [live_price],
            'High': [live_price],
            'Low': [live_price],
            'Open': [live_price],
            'Volume': [0]
        })
        live_data.set_index('Date', inplace=True)
        history = pd.concat([history, live_data])

    return history, stock, live_price, pre_post_price

# Calculate technical indicators
def calculate_indicators(data):
    data['MA10'] = data['Close'].rolling(10).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    delta = data['Close'].diff(1)
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    data['RSI'] = 100 - (100 / (1 + (gain / loss)))
    data['ATR'] = (data['High'] - data['Low']).rolling(14).mean()
    data['EMA12'] = data['Close'].ewm(span=12).mean()
    data['EMA26'] = data['Close'].ewm(span=26).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    return data

# Detect support and resistance
def detect_support_resistance(data):
    local_max = argrelextrema(data['Close'].values, np.greater, order=5)[0]
    local_min = argrelextrema(data['Close'].values, np.less, order=5)[0]
    support_levels = sorted(data['Close'].iloc[local_min].tail(3).values)
    resistance_levels = sorted(data['Close'].iloc[local_max].tail(3).values, reverse=True)
    return support_levels, resistance_levels

# Predict next week's trend using ARIMA
def predict_next_week(data):
    try:
        close_data = data['Close'].dropna().reset_index(drop=True)
        if len(close_data) < 20:
            return [close_data.iloc[-1]] * 7
        model = ARIMA(close_data, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=7)
        return forecast.values
    except Exception:
        return [data['Close'].iloc[-1]] * 7

# Fetch options chain
def fetch_options_chain(stock, live_price):
    expiry_dates = stock.options
    options = stock.option_chain(expiry_dates[0])
    calls = options.calls
    puts = options.puts

    lower_bound = live_price * 0.95
    upper_bound = live_price * 1.05
    calls = calls[(calls['strike'] >= lower_bound) & (calls['strike'] <= upper_bound)].nlargest(5, 'volume')
    puts = puts[(puts['strike'] >= lower_bound) & (puts['strike'] <= upper_bound)].nlargest(5, 'volume')

    calls['Expiration'] = expiry_dates[0]
    puts['Expiration'] = expiry_dates[0]
    return calls, puts

# Streamlit App
st.title("💰 Let’s Get That Money! 🚀")
st.caption("The ultimate AI-powered stock analysis tool with candlestick trends, indicators, volume, and next week predictions.")

ticker_input = st.text_input("Enter Stock Ticker (e.g., TSLA, SPY):", "")
if ticker_input:
    data, stock, live_price, pre_post_price = get_stock_data(ticker_input.upper())
    if not data.empty:
        data = calculate_indicators(data)
        support_levels, resistance_levels = detect_support_resistance(data)
        predicted_trend = predict_next_week(data)
        next_week_dates = pd.date_range(start=datetime.datetime.now(), periods=7)
        calls, puts = fetch_options_chain(stock, live_price)

        # Trade Signal Summary
        st.write("## Trade Signal Summary")
        st.table(pd.DataFrame({
            "Live Price": [f"${live_price:.2f} (Pre/Post: ${pre_post_price:.2f})" if pre_post_price else f"${live_price:.2f}"],
            "Next Week Trend (Avg)": [f"${np.mean(predicted_trend):.2f}"],
            "Signal": ["UPTREND" if predicted_trend[-1] > live_price else "DOWNTREND"],
            "Suggested Strike": [f"${round(np.mean(predicted_trend), 2)}"]
        }))

        # Stock Chart
        st.write("### Stock Chart with Indicators and Next Week Trend")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                                     low=data['Low'], close=data['Close'], name="Candlestick"))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA10'], name="MA10", line=dict(color="orange", dash="dot")))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], name="MA50", line=dict(color="red", dash="dot")))
        fig.add_trace(go.Scatter(x=next_week_dates, y=predicted_trend, name="Next Week Trend", line=dict(color="green", dash="dash")))
        st.plotly_chart(fig, use_container_width=True)

        # Top Call Options
        st.write("### Top Call Options")
        st.table(calls[['strike', 'lastPrice', 'volume', 'openInterest', 'Expiration', 'lastTradeDate']])

        # Top Put Options
        st.write("### Top Put Options")
        st.table(puts[['strike', 'lastPrice', 'volume', 'openInterest', 'Expiration', 'lastTradeDate']])

        # Support/Resistance Table
        st.write("### Support and Resistance Levels")
        st.table(pd.DataFrame({"Support": support_levels, "Resistance": resistance_levels}))

        # Fibonacci Table
        st.write("### Fibonacci Retracement and Extension Levels")
        retracements, extensions = calculate_fibonacci(data)
        combined_fib = {**retracements, **extensions}
        st.table(pd.DataFrame(combined_fib.items(), columns=["Level", "Price"]))
