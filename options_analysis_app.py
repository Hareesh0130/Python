import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from statsmodels.tsa.arima.model import ARIMA
from textblob import TextBlob
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import requests

# Fetch stock data
def get_stock_data(symbol, interval='1d', days=90):
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=days)
    stock = yf.Ticker(symbol)
    return stock.history(interval=interval, start=start_date, end=end_date), stock

# Calculate technical indicators
def calculate_indicators(data):
    data['MA10'] = data['Close'].rolling(10).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    delta = data['Close'].diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['ATR'] = (data['High'] - data['Low']).rolling(14).mean()
    data['Upper_BB'] = data['Close'].rolling(20).mean() + 2 * data['Close'].rolling(20).std()
    data['Lower_BB'] = data['Close'].rolling(20).mean() - 2 * data['Close'].rolling(20).std()
    data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    return data

# Detect support and resistance
def detect_support_resistance(data):
    local_max = argrelextrema(data['Close'].values, np.greater, order=5)[0]
    local_min = argrelextrema(data['Close'].values, np.less, order=5)[0]
    support_levels = sorted(data['Close'].iloc[local_min].tail(3).values)
    resistance_levels = sorted(data['Close'].iloc[local_max].tail(3).values, reverse=True)
    return support_levels, resistance_levels

# Calculate Fibonacci retracement levels
def calculate_fibonacci(data):
    high = data['Close'].max()
    low = data['Close'].min()
    diff = high - low
    return {
        '0.0%': low,
        '23.6%': low + 0.236 * diff,
        '38.2%': low + 0.382 * diff,
        '50.0%': low + 0.5 * diff,
        '61.8%': low + 0.618 * diff,
        '100.0%': high
    }

# Predict price using XGBoost
def predict_with_xgboost(data):
    features = ['MA10', 'MA50', 'RSI', 'ATR', 'MACD', 'MACD_Signal']
    data = data.dropna()
    X = data[features]
    y = data['Close']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = XGBRegressor()
    model.fit(X_train, y_train)
    next_day_features = X_scaled[-1].reshape(1, -1)
    prediction = model.predict(next_day_features)
    return prediction[0]

# Generate trade signals
def generate_trade_signal(data, predicted_price):
    live_price = data['Close'].iloc[-1]
    rsi = data['RSI'].iloc[-1]
    macd = data['MACD'].iloc[-1]
    macd_signal = data['MACD_Signal'].iloc[-1]
    atr = data['ATR'].iloc[-1]

    if predicted_price > live_price and macd > macd_signal and rsi < 70:
        signal = "BUY CALL"
    elif predicted_price < live_price and macd < macd_signal and rsi > 70:
        signal = "BUY PUT"
    else:
        signal = "HOLD"

    stop_loss = live_price - (1.5 * atr) if signal == "BUY CALL" else live_price + (1.5 * atr)
    take_profit = live_price + (2 * atr) if signal == "BUY CALL" else live_price - (2 * atr)
    return signal, stop_loss, take_profit

# Streamlit App
st.title("💰 Let’s Get That Money! 🚀")
st.caption("AI predictions, signals, Fibonacci levels, and more – all in one place.")

ticker_input = st.text_input("Enter Stock Tickers (comma-separated):", "")
if ticker_input:
    tickers = [ticker.strip().upper() for ticker in ticker_input.split(",")]
    for ticker in tickers:
        data, stock = get_stock_data(ticker)
        if not data.empty:
            current_price = data['Close'].iloc[-1]
            data = calculate_indicators(data)
            support_levels, resistance_levels = detect_support_resistance(data)
            fib_levels = calculate_fibonacci(data)
            predicted_price = predict_with_xgboost(data)
            signal, stop_loss, take_profit = generate_trade_signal(data, predicted_price)

            # Display Trade Signal
            st.write(f"## Trade Signal for {ticker}")
            st.table(pd.DataFrame({
                "Current Price": [f"${current_price:.2f}"],
                "Signal": [signal],
                "Predicted Price": [f"${predicted_price:.2f}"],
                "Stop Loss": [f"${stop_loss:.2f}"],
                "Take Profit": [f"${take_profit:.2f}"],
            }))

            # Enhanced Chart
            st.write("### Stock Chart with Indicators")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close Price", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=data.index, y=data['MA10'], name="MA10", line=dict(color="orange", dash="dot")))
            fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], name="MA50", line=dict(color="red", dash="dot")))
            fig.add_trace(go.Scatter(x=data.index, y=data['Upper_BB'], name="Upper BB", line=dict(color="purple", dash="dot")))
            fig.add_trace(go.Scatter(x=data.index, y=data['Lower_BB'], name="Lower BB", line=dict(color="purple", dash="dot")))

            # Add Support and Resistance Lines
            for level in support_levels:
                fig.add_shape(type="line", y0=level, y1=level, x0=data.index[0], x1=data.index[-1],
                              line=dict(color="green", dash="dash"))
            for level in resistance_levels:
                fig.add_shape(type="line", y0=level, y1=level, x0=data.index[0], x1=data.index[-1],
                              line=dict(color="red", dash="dash"))

            st.plotly_chart(fig, use_container_width=True)

            # Display Fibonacci Levels
            st.write("### Fibonacci Levels")
            st.table(pd.DataFrame(fib_levels.items(), columns=["Level", "Price"]))

            # Support and Resistance
            st.write("### Support and Resistance")
            st.table(pd.DataFrame({"Support": support_levels, "Resistance": resistance_levels}))
