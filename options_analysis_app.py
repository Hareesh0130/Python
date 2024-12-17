import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema

# Function to fetch stock data
def get_stock_data(symbol, interval='1d', days=90):
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=days)
    stock = yf.Ticker(symbol)
    return stock.history(interval=interval, start=start_date, end=end_date), stock

# Function to calculate indicators
def calculate_indicators(data):
    data['MA10'] = data['Close'].rolling(10, min_periods=1).mean()
    data['MA50'] = data['Close'].rolling(50, min_periods=1).mean()

    # RSI
    delta = data['Close'].diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # ATR
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift(1))
    low_close = abs(data['Low'] - data['Close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = true_range.rolling(14, min_periods=1).mean()

    # MACD
    data['EMA12'] = data['Close'].ewm(span=12).mean()
    data['EMA26'] = data['Close'].ewm(span=26).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()

    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    data['BB_Upper'] = data['BB_Middle'] + 2 * data['Close'].rolling(window=20).std()
    data['BB_Lower'] = data['BB_Middle'] - 2 * data['Close'].rolling(window=20).std()

    # VWAP
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()

    # Relative Volume
    data['RVOL'] = data['Volume'] / data['Volume'].rolling(20).mean()

    return data

# Function to detect trends and key levels
def detect_trends(data):
    local_max = argrelextrema(data['Close'].values, np.greater, order=5)[0]
    local_min = argrelextrema(data['Close'].values, np.less, order=5)[0]

    trend = "Sideways"
    if len(local_max) > 1 and len(local_min) > 1:
        last_high = data['Close'].iloc[local_max[-1]]
        prev_high = data['Close'].iloc[local_max[-2]]
        last_low = data['Close'].iloc[local_min[-1]]
        prev_low = data['Close'].iloc[local_min[-2]]

        if last_high > prev_high and last_low > prev_low:
            trend = "Uptrend"
        elif last_high < prev_high and last_low < prev_low:
            trend = "Downtrend"

    return local_max, local_min, trend

# Recommendation System
def generate_recommendation(data, trend, support, resistance):
    live_price = data['Close'].iloc[-1]
    rsi = data['RSI'].iloc[-1]
    macd = data['MACD'].iloc[-1]
    macd_signal = data['MACD_Signal'].iloc[-1]
    atr = data['ATR'].iloc[-1]
    rvol = data['RVOL'].iloc[-1]

    score = 0
    signals = []

    if trend == "Uptrend":
        score += 2
        signals.append("Trend: Uptrend (+2)")
    elif trend == "Downtrend":
        score -= 2
        signals.append("Trend: Downtrend (-2)")

    if rsi < 30:
        score += 2
        signals.append("RSI: Oversold (+2)")
    elif rsi > 70:
        score -= 2
        signals.append("RSI: Overbought (-2)")

    if macd > macd_signal:
        score += 1
        signals.append("MACD: Bullish Crossover (+1)")
    elif macd < macd_signal:
        score -= 1
        signals.append("MACD: Bearish Crossover (-1)")

    if live_price <= support + atr:
        score += 1
        signals.append("Near Support (+1)")
    elif live_price >= resistance - atr:
        score -= 1
        signals.append("Near Resistance (-1)")

    if rvol > 1.5:
        score += 1
        signals.append("Relative Volume: High (+1)")

    recommendation = "Hold"
    if score >= 5:
        recommendation = "Strong Buy"
    elif 2 <= score < 5:
        recommendation = "Buy"
    elif -4 <= score < -2:
        recommendation = "Sell"
    elif score <= -4:
        recommendation = "Strong Sell"

    return recommendation, signals

# Fetch options chain
def fetch_options_chain(symbol, current_price):
    stock = yf.Ticker(symbol)
    expiry_dates = stock.options
    if not expiry_dates:
        return None, None

    options_chain = stock.option_chain(expiry_dates[0])
    calls = options_chain.calls.copy()
    puts = options_chain.puts.copy()

    calls['Intrinsic'] = (current_price - calls['strike']).clip(lower=0)
    puts['Intrinsic'] = (puts['strike'] - current_price).clip(lower=0)

    calls = calls.sort_values(by=['volume', 'openInterest'], ascending=False).head(3)
    puts = puts.sort_values(by=['volume', 'openInterest'], ascending=False).head(3)
    return calls, puts

# Streamlit App
st.title("Comprehensive Stock Analysis with Options and Recommendations")

ticker_input = st.text_input("Enter Stock Tickers (comma-separated, e.g., TSLA, AAPL):", "")
if ticker_input:
    tickers = [ticker.strip().upper() for ticker in ticker_input.split(",")]
    summary_data = []

    for ticker in tickers:
        data, stock = get_stock_data(ticker)
        if not data.empty:
            data = calculate_indicators(data)
            local_max, local_min, trend = detect_trends(data)
            live_price = data['Close'].iloc[-1]
            support = data['Close'].iloc[local_min].min() if len(local_min) > 0 else "N/A"
            resistance = data['Close'].iloc[local_max].max() if len(local_max) > 0 else "N/A"

            recommendation, signals = generate_recommendation(data, trend, support, resistance)
            calls, puts = fetch_options_chain(ticker, live_price)

            summary_data.append({
                "Ticker": ticker,
                "Live Price": f"${live_price:.2f}",
                "Trend": trend,
                "Support": f"${support:.2f}",
                "Resistance": f"${resistance:.2f}",
                "RSI": f"{data['RSI'].iloc[-1]:.2f}",
                "ATR": f"${data['ATR'].iloc[-1]:.2f}",
                "Recommendation": recommendation
            })

            # Display Top Call and Put Options and Chart
            st.write(f"### Detailed Analysis for {ticker}")
            st.write("#### Recommendation Signals")
            for signal in signals:
                st.write(f"- {signal}")

            st.write("#### Stock Chart")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close Price"))
            fig.add_trace(go.Scatter(x=data.index, y=data['MA10'], name="MA10", line=dict(dash="dot")))
            fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], name="MA50", line=dict(dash="dot")))
            st.plotly_chart(fig)

            st.write("#### Top Call Options")
            st.dataframe(calls)

            st.write("#### Top Put Options")
            st.dataframe(puts)

    # Display Summary Table at the Top
    st.write("## Summary Table")
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df)
