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
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    delta = data['Close'].diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
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

# Predict price using Machine Learning
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

# Fetch options chain
def fetch_options_chain(stock, live_price):
    options = stock.option_chain()
    calls = options.calls
    puts = options.puts
    expiry_date = stock.options[0]
    lower_bound = live_price * 0.95
    upper_bound = live_price * 1.05
    calls = calls[(calls['strike'] >= lower_bound) & (calls['strike'] <= upper_bound)]
    puts = puts[(puts['strike'] >= lower_bound) & (puts['strike'] <= upper_bound)]
    return calls, puts, expiry_date

# Sentiment analysis
def fetch_sentiment(ticker):
    api_key = "YOUR_NEWS_API_KEY"
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}"
    response = requests.get(url).json()
    headlines = [article['title'] for article in response.get('articles', [])[:5]]
    sentiment_score = sum([TextBlob(headline).sentiment.polarity for headline in headlines])
    return "Positive" if sentiment_score > 0 else "Negative"

# Trade signal generator
def generate_trade_signal(data, calls, puts, predicted_price):
    live_price = data['Close'].iloc[-1]
    rsi = data['RSI'].iloc[-1]
    macd = data['MACD'].iloc[-1]
    macd_signal = data['MACD_Signal'].iloc[-1]
    atr = data['ATR'].iloc[-1]

    signal = "HOLD"
    strike_price = None
    stop_loss = take_profit = None

    if predicted_price > live_price and macd > macd_signal and rsi < 70:
        signal = "BUY CALL"
        strike_price = calls.iloc[0]['strike'] if not calls.empty else "N/A"
        stop_loss = live_price - (1.5 * atr)
        take_profit = live_price + (2 * atr)

    elif predicted_price < live_price and macd < macd_signal and rsi > 70:
        signal = "BUY PUT"
        strike_price = puts.iloc[0]['strike'] if not puts.empty else "N/A"
        stop_loss = live_price + (1.5 * atr)
        take_profit = live_price - (2 * atr)

    return signal, strike_price, stop_loss, take_profit

# Streamlit App
st.title("💰 Let’s Get That Money! 🚀")
st.caption("Stock signals, AI predictions, Fibonacci levels, and more – all in one place.")

ticker_input = st.text_input("Enter Stock Tickers (comma-separated):", "")
if ticker_input:
    tickers = [ticker.strip().upper() for ticker in ticker_input.split(",")]
    for ticker in tickers:
        data, stock = get_stock_data(ticker)
        if not data.empty:
            data = calculate_indicators(data)
            support_levels, resistance_levels = detect_support_resistance(data)
            fib_levels = calculate_fibonacci(data)
            predicted_price = predict_with_xgboost(data)
            calls, puts, expiry_date = fetch_options_chain(stock, data['Close'].iloc[-1])
            signal, strike_price, stop_loss, take_profit = generate_trade_signal(data, calls, puts, predicted_price)
            sentiment = fetch_sentiment(ticker)

            # Display Trade Signal
            st.write(f"## Trade Signal for {ticker}")
            st.table(pd.DataFrame({
                "Signal": [signal],
                "Predicted Price": [f"${predicted_price:.2f}"],
                "Strike Price": [strike_price],
                "Stop Loss": [f"${stop_loss:.2f}" if stop_loss else "N/A"],
                "Take Profit": [f"${take_profit:.2f}" if take_profit else "N/A"],
                "Expiry Date": [expiry_date],
                "Sentiment": [sentiment]
            }))

            # Display Stock Chart
            st.write("### Stock Chart")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close Price"))
            for level in fib_levels.values():
                fig.add_shape(type="line", y0=level, y1=level, x0=data.index[0], x1=data.index[-1], line=dict(color="purple", dash="dot"))
            st.plotly_chart(fig, use_container_width=True)

            # Display Fibonacci Levels
            st.write("### Fibonacci Levels")
            st.table(pd.DataFrame(fib_levels.items(), columns=["Level", "Price"]))

            # Support and Resistance
            st.write("### Support and Resistance")
            st.table(pd.DataFrame({"Support": support_levels, "Resistance": resistance_levels}))

            # Options Chain
            st.write("### Top Call Options")
            st.dataframe(calls)

            st.write("### Top Put Options")
            st.dataframe(puts)
