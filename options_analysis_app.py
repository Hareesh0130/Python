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

# Detect Buy and Sell Signals
def detect_buy_sell_signals(data, predicted_price):
    buy_signals = [np.nan] * len(data)
    sell_signals = [np.nan] * len(data)

    for i in range(1, len(data)):
        # Buy Now Condition
        if data['RSI'].iloc[i] < 30 and data['MACD'].iloc[i] > data['MACD_Signal'].iloc[i] and predicted_price > data['Close'].iloc[i]:
            buy_signals[i] = data['Close'].iloc[i]
        # Sell Now Condition
        elif data['RSI'].iloc[i] > 70 and data['MACD'].iloc[i] < data['MACD_Signal'].iloc[i] and predicted_price < data['Close'].iloc[i]:
            sell_signals[i] = data['Close'].iloc[i]

    data['Buy_Now'] = buy_signals
    data['Sell_Now'] = sell_signals
    return data

# Hybrid prediction: XGBoost + ARIMA
def hybrid_prediction(data):
    features = ['MA10', 'MA50', 'RSI', 'ATR', 'MACD', 'MACD_Signal']
    data = data.dropna()
    X = data[features]
    y = data['Close']

    # Scale features and train XGBoost
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = XGBRegressor()
    model.fit(X_train, y_train)
    xgb_pred = model.predict(X_scaled[-1].reshape(1, -1))[0]

    # ARIMA fallback
    close_data = data['Close'].dropna().reset_index(drop=True)
    arima_model = ARIMA(close_data, order=(5, 1, 0))
    arima_pred = arima_model.fit().forecast(steps=1)[0]

    return round((xgb_pred + arima_pred) / 2, 2)

# Streamlit App
st.title("💰 Let’s Get That Money! 🚀")
st.caption("AI predictions, trade signals, Fibonacci levels, and 'Buy Now'/'Sell Now' indicators.")

ticker_input = st.text_input("Enter Stock Ticker (e.g., TSLA):", "")
if ticker_input:
    data, stock = get_stock_data(ticker_input.upper())
    if not data.empty:
        current_price = data['Close'].iloc[-1]
        data = calculate_indicators(data)
        support_levels, resistance_levels = detect_support_resistance(data)
        predicted_price = hybrid_prediction(data)
        data = detect_buy_sell_signals(data, predicted_price)

        # Summary Table
        st.write("## Trade Signal Summary")
        st.table(pd.DataFrame({
            "Current Price": [f"${current_price:.2f}"],
            "Predicted Price": [f"${predicted_price:.2f}"],
            "Signal": ["BUY CALL" if predicted_price > resistance_levels[0] else "BUY PUT" if predicted_price < support_levels[0] else "HOLD"],
            "Reason": ["Predicted price exceeds resistance" if predicted_price > resistance_levels[0] else
                       "Predicted price falls below support" if predicted_price < support_levels[0] else "Price within range"]
        }))

        # Enhanced Chart with Buy Now/Sell Now Indicators
        st.write("### Stock Chart with Indicators and Signals")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close Price", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA10'], name="MA10", line=dict(color="orange", dash="dot")))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], name="MA50", line=dict(color="red", dash="dot")))
        
        # Add Buy Now and Sell Now Markers
        fig.add_trace(go.Scatter(x=data.index, y=data['Buy_Now'], mode='markers', name="Buy Now",
                                 marker=dict(color="green", size=10, symbol="triangle-up")))
        fig.add_trace(go.Scatter(x=data.index, y=data['Sell_Now'], mode='markers', name="Sell Now",
                                 marker=dict(color="red", size=10, symbol="triangle-down")))
        
        # Add Support and Resistance Lines
        for support in support_levels:
            fig.add_shape(type="line", y0=support, y1=support, x0=data.index[0], x1=data.index[-1],
                          line=dict(color="green", dash="dash"))
        for resistance in resistance_levels:
            fig.add_shape(type="line", y0=resistance, y1=resistance, x0=data.index[0], x1=data.index[-1],
                          line=dict(color="red", dash="dash"))
        
        st.plotly_chart(fig, use_container_width=True)
