import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

def get_stock_data(symbol, interval='1d', days=90):
    """
    Fetch historical stock data.
    """
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=days)
    stock = yf.Ticker(symbol)
    return stock.history(interval=interval, start=start_date, end=end_date)

def calculate_indicators(data):
    """
    Calculate technical indicators: MA, RSI, ATR, VWAP.
    """
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
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = tr.rolling(14, min_periods=1).mean()

    # VWAP
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()

    return data

def detect_patterns(data):
    """
    Detect chart patterns: Head & Shoulders, Double Top/Bottom.
    """
    local_max = argrelextrema(data['Close'].values, np.greater, order=5)[0]
    local_min = argrelextrema(data['Close'].values, np.less, order=5)[0]
    patterns = {'Double Top': False, 'Double Bottom': False}

    # Double Top
    if len(local_max) >= 2 and abs(data['Close'].iloc[local_max[-1]] - data['Close'].iloc[local_max[-2]]) < 1e-2:
        patterns['Double Top'] = True
    # Double Bottom
    if len(local_min) >= 2 and abs(data['Close'].iloc[local_min[-1]] - data['Close'].iloc[local_min[-2]]) < 1e-2:
        patterns['Double Bottom'] = True

    return patterns, local_max, local_min

def fetch_options_chain(symbol):
    """
    Fetch options chain for the next expiry and recommend top actively bought calls and puts.
    """
    stock = yf.Ticker(symbol)
    expiry_dates = stock.options
    if not expiry_dates:
        return None, None

    options_chain = stock.option_chain(expiry_dates[0])
    calls = options_chain.calls.copy()
    puts = options_chain.puts.copy()

    # Calculate Volume-to-Open Interest Ratio for Calls and Puts
    calls['Volume_OI_Ratio'] = calls['volume'] / (calls['openInterest'] + 1)  # Avoid division by zero
    puts['Volume_OI_Ratio'] = puts['volume'] / (puts['openInterest'] + 1)

    # Sort Calls and Puts by Volume-to-OI Ratio and Volume
    calls = calls.sort_values(by=['Volume_OI_Ratio', 'volume'], ascending=[False, False]).head(3)
    puts = puts.sort_values(by=['Volume_OI_Ratio', 'volume'], ascending=[False, False]).head(3)

    return calls, puts


def generate_recommendation(data, patterns):
    """
    Generate buy/sell recommendations based on patterns and indicators.
    """
    live_price = data['Close'].iloc[-1]
    atr = data['ATR'].iloc[-1]
    support = data['Close'].min()
    resistance = data['Close'].max()

    recommendation = "Hold"
    if patterns['Double Bottom']:
        recommendation = "Strong Buy"
    elif patterns['Double Top']:
        recommendation = "Strong Sell"

    stop_loss = live_price - 2 * atr
    take_profit = live_price + 4 * atr
    return recommendation, stop_loss, take_profit, support, resistance

def visualize_chart(data, local_max, local_min, symbol, support, resistance):
    """
    Visualize stock data, indicators, and support/resistance levels.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Close Price', color='black')
    plt.plot(data['MA10'], label='MA10', linestyle='--', color='blue')
    plt.plot(data['MA50'], label='MA50', linestyle='--', color='red')

    # Support and Resistance Triangles
    plt.scatter(data.index[local_max], data['Close'].iloc[local_max], color='red', label='Resistance', marker='^')
    plt.scatter(data.index[local_min], data['Close'].iloc[local_min], color='green', label='Support', marker='v')

    # Horizontal Lines for Support and Resistance
    plt.axhline(resistance, color='red', linestyle='--', label=f"Resistance: {resistance:.2f}")
    plt.axhline(support, color='green', linestyle='--', label=f"Support: {support:.2f}")

    plt.title(f"{symbol} - Chart Patterns and Indicators")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    print("Multiple Ticker Analysis with Chart Patterns and Options Recommendations")
    symbols = input("Enter stock symbols (comma-separated, e.g., TSLA, AAPL, MSFT): ").upper().split(",")

    for symbol in symbols:
        symbol = symbol.strip()
        print(f"\n{'='*20} Analyzing {symbol} {'='*20}")
        
        # Fetch stock data
        data = get_stock_data(symbol, days=90)
        if data.empty:
            print("No data found. Skipping.")
            continue

        # Calculate indicators and detect patterns
        data = calculate_indicators(data)
        patterns, local_max, local_min = detect_patterns(data)
        calls, puts = fetch_options_chain(symbol)

        # Generate recommendations
        recommendation, stop_loss, take_profit, support, resistance = generate_recommendation(data, patterns)

        # Display results
        print(f"Recommendation: {recommendation}")
        print(f"Live Price: ${data['Close'].iloc[-1]:.2f}")
        print(f"Stop Loss: ${stop_loss:.2f}, Take Profit: ${take_profit:.2f}")
        print(f"Support: ${support:.2f}, Resistance: ${resistance:.2f}")
        print("Detected Patterns:")
        for pattern, detected in patterns.items():
            print(f"- {pattern}: {'Yes' if detected else 'No'}")

        # Display options chain
        if calls is not None and puts is not None:
            print("\nTop Call Options:")
            print(calls[['strike', 'lastPrice', 'openInterest', 'volume']])

            print("\nTop Put Options:")
            print(puts[['strike', 'lastPrice', 'openInterest', 'volume']])
        else:
            print("No options data available for this stock.")

        # Visualize the chart
        visualize_chart(data, local_max, local_min, symbol, support, resistance)

if __name__ == "__main__":
    main()
