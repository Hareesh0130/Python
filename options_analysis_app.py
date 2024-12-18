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

# Fetch stock data with pre/post market price
def get_stock_data(symbol, interval='1d', days=180):
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=days)
    stock = yf.Ticker(symbol)
    history = stock.history(interval=interval, start=start_date, end=end_date)
    live_price = stock.info.get("currentPrice", None)
    pre_post_price = stock.info.get("preMarketPrice", stock.info.get("postMarketPrice", None))
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
    data['BB_Mid'] = data['Close'].rolling(20).mean()
    data['BB_Upper'] = data['BB_Mid'] + 2 * data['Close'].rolling(20).std()
    data['BB_Lower'] = data['BB_Mid'] - 2 * data['Close'].rolling(20).std()
    return data

# Detect support and resistance
def detect_support_resistance(data):
    local_max = argrelextrema(data['Close'].values, np.greater, order=5)[0]
    local_min = argrelextrema(data['Close'].values, np.less, order=5)[0]
    support_levels = sorted(data['Close'].iloc[local_min].tail(3).values)
    resistance_levels = sorted(data['Close'].iloc[local_max].tail(3).values, reverse=True)
    return support_levels, resistance_levels

# Calculate Fibonacci retracement and extension levels
def calculate_fibonacci(data):
    high = data['Close'].max()
    low = data['Close'].min()
    diff = high - low
    retracements = {
        '0.0%': low,
        '23.6%': low + 0.236 * diff,
        '38.2%': low + 0.382 * diff,
        '50.0%': low + 0.5 * diff,
        '61.8%': low + 0.618 * diff,
        '100.0%': high
    }
    extensions = {
        '161.8%': high + 0.618 * diff,
        '200.0%': high + diff,
        '261.8%': high + 1.618 * diff
    }
    return retracements, extensions

# Fetch options chain data with expiration
def fetch_options_chain(stock, live_price):
    expiry_dates = stock.options
    options = stock.option_chain(expiry_dates[0])  # Fetch first expiration date
    calls = options.calls
    puts = options.puts

    lower_bound = live_price * 0.95
    upper_bound = live_price * 1.05
    calls = calls[(calls['strike'] >= lower_bound) & (calls['strike'] <= upper_bound)].nlargest(5, 'volume')
    puts = puts[(puts['strike'] >= lower_bound) & (puts['strike'] <= upper_bound)].nlargest(5, 'volume')

    # Add expiration date column
    calls['Expiration'] = expiry_dates[0]
    puts['Expiration'] = expiry_dates[0]

    return calls, puts, expiry_dates[0]

# ARIMA fallback prediction
def predict_with_arima(data):
    try:
        close_data = data['Close'].dropna().reset_index(drop=True)
        if len(close_data) < 15:
            return close_data.iloc[-1]
        model = ARIMA(close_data, order=(5, 1, 0))
        model_fit = model.fit()
        return round(model_fit.forecast(steps=1)[0], 2)
    except Exception:
        return round(data['Close'].iloc[-1], 2)

# Hybrid prediction: XGBoost + ARIMA
def hybrid_prediction(data):
    features = ['MA10', 'MA50', 'RSI', 'ATR', 'MACD', 'MACD_Signal']
    data = data.dropna()
    X = data[features]
    y = data['Close']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = XGBRegressor()
    model.fit(X_train, y_train)
    xgb_pred = model.predict(X_scaled[-1].reshape(1, -1))[0]

    arima_pred = predict_with_arima(data)
    return round((xgb_pred + arima_pred) / 2, 2)

# Streamlit App
st.title("💰 Let’s Get That Money! 🚀")
st.caption("AI predictions, Fibonacci levels, options chain, support/resistance, and trade signals.")

ticker_input = st.text_input("Enter Stock Ticker (e.g., TSLA):", "")
if ticker_input:
    data, stock, live_price, pre_post_price = get_stock_data(ticker_input.upper())
    if not data.empty:
        current_price = data['Close'].iloc[-1]
        data = calculate_indicators(data)
        support_levels, resistance_levels = detect_support_resistance(data)
        retracements, extensions = calculate_fibonacci(data)
        predicted_price = hybrid_prediction(data)
        calls, puts, expiry_date = fetch_options_chain(stock, live_price)
        pips_moved = round(abs(predicted_price - current_price), 2)

        # Summary Table
        st.write("## Trade Signal Summary")
        st.table(pd.DataFrame({
            "Current Price": [f"${current_price:.2f}"],
            "Live Price": [f"${live_price:.2f} (Pre/Post: ${pre_post_price:.2f})" if pre_post_price else f"${live_price:.2f}"],
            "Predicted Price": [f"${predicted_price:.2f}"],
            "Pips Moved": [f"{pips_moved}"],
            "Signal": ["BUY CALL" if predicted_price > resistance_levels[0] else "BUY PUT" if predicted_price < support_levels[0] else "HOLD"],
            "Suggested Strike": [f"${round(predicted_price, 2)}"],
            "Call Expiry": [expiry_date]
        }))

        # Options Chain Tables
        st.write("### Top Call Options")
        st.table(calls[['strike', 'lastPrice', 'volume', 'openInterest', 'Expiration', 'lastTradeDate']])

        st.write("### Top Put Options")
        st.table(puts[['strike', 'lastPrice', 'volume', 'openInterest', 'Expiration', 'lastTradeDate']])

        # Enhanced Chart
        st.write("### Stock Chart with Indicators")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close Price", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA10'], name="MA10", line=dict(color="orange", dash="dot")))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], name="MA50", line=dict(color="red", dash="dot")))
        for level in support_levels:
            fig.add_shape(type="line", y0=level, y1=level, x0=data.index[0], x1=data.index[-1], line=dict(color="green", dash="dash"))
        for level in resistance_levels:
            fig.add_shape(type="line", y0=level, y1=level, x0=data.index[0], x1=data.index[-1], line=dict(color="red", dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

        # Fibonacci Table
        st.write("### Fibonacci Retracement and Extension Levels")
        combined_fib = {**retracements, **extensions}
        st.table(pd.DataFrame(combined_fib.items(), columns=["Level", "Price"]))

        # Support/Resistance Table
        st.write("### Support and Resistance Levels")
        st.table(pd.DataFrame({"Support": support_levels, "Resistance": resistance_levels}))
