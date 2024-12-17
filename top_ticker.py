import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from finvizfinance.screener.overview import Overview
from twilio.rest import Client

# Twilio credentials
ACCOUNT_SID = "AC18d676e135186160c538f0391a26973e"
AUTH_TOKEN = "5b0ec3cc11afc2e2acac96ad7547ac1c"
TWILIO_PHONE = "+18885694951"
YOUR_PHONE = "+12168011676"  # Replace with your phone number

# Fetching top moving tickers
def get_top_movers():
    """
    Fetch top gainers from Finviz dynamically.
    Returns:
        list: A list of top ticker symbols sorted by percentage gain.
    """
    overview = Overview()
    overview.set_filter(signal='Top Gainers')  # Use 'Top Gainers' filter
    data = overview.screener_view()
    data = data.sort_values(by='Change', ascending=False)  # Sort by percentage change (top gainers)
    return data['Ticker'].tolist()

def get_stock_data(symbol, interval='1h', days=30):
    """
    Fetch historical stock data for a given symbol with hourly interval (includes pre/post market).

    Args:
        symbol (str): Stock ticker symbol.
        interval (str): Data interval (e.g., '1h' for hourly).
        days (int): Number of days of historical data.

    Returns:
        DataFrame: Historical stock data.
    """
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    stock = yf.Ticker(symbol)
    data = stock.history(interval=interval, start=start_date, end=end_date)
    return data

def calculate_moving_averages(data):
    """
    Calculate 10-day, 14-day, and 50-day moving averages.
    """
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA14'] = data['Close'].rolling(window=14).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    return data

def calculate_rsi(data, period=14):
    """
    Calculate the Relative Strength Index (RSI).
    """
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def calculate_atr(data, period=14):
    """
    Calculate the Average True Range (ATR) for dynamic stop-loss.
    """
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift(1))
    low_close = abs(data['Low'] - data['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = tr.rolling(window=period).mean()
    return data

def find_support_resistance(data):
    """
    Find support and resistance levels.
    """
    support = data['Close'].min()
    resistance = data['Close'].max()
    return support, resistance

def send_sms(recommendations):
    """
    Send stock recommendations via SMS using Twilio.
    """
    client = Client(ACCOUNT_SID, AUTH_TOKEN)

    message_body = "Here are pre-market top movers:\n\n"
    for rec in recommendations:
        message_body += (
            f"Trading Levels for {rec['symbol']}:\n"
            f"Current Price: ${rec['close_price']:.2f}\n"
            f"Stop Loss: ${rec['stop_loss']:.2f} (Dynamic via ATR)\n"
            f"Target Profit: ${rec['target_profit']:.2f}\n"
            f"Support Level: ${rec['support']:.2f}, Resistance Level: ${rec['resistance']:.2f}\n"
            f"Recommendation: {rec['recommendation']}\n"
            f"Suggested Call Option Expiry: {datetime.date.today() + datetime.timedelta(days=7)}\n\n"
        )

    message = client.messages.create(
        body=message_body,
        from_=TWILIO_PHONE,
        to=YOUR_PHONE
    )

    print(f"Message sent! SID: {message.sid}")

def provide_trading_levels(data, symbol):
    """
    Provide trading levels and recommendations using ATR, RSI, and moving averages.
    """
    close_price = data['Close'].iloc[-1]
    support, resistance = find_support_resistance(data)
    rsi = data['RSI'].iloc[-1]
    atr = data['ATR'].iloc[-1]

    stop_loss = close_price - atr
    target_profit = close_price + (atr * 2)
    ma10, ma14, ma50 = data['MA10'].iloc[-1], data['MA14'].iloc[-1], data['MA50'].iloc[-1]

    recommendation = "Hold"
    if close_price > ma10 and close_price > ma14 and rsi < 70:
        recommendation = "Buy Now"
    elif rsi > 70:
        recommendation = "Sell Now - Overbought"
    elif close_price < support + atr and rsi < 30:
        recommendation = "Strong Buy - Oversold"

    return {
        "symbol": symbol,
        "close_price": close_price,
        "stop_loss": stop_loss,
        "target_profit": target_profit,
        "support": support,
        "resistance": resistance,
        "recommendation": recommendation
    }

def main():
    print("Stock Trading Advisor with Live Indicators")

    # Get top movers
    print("Fetching top moving tickers...")
    tickers = get_top_movers()[:5]  # Only consider the top 5 gainers
    print(f"Top 5 movers: {', '.join(tickers)}\n")

    recommendations = []

    for symbol in tickers:  # Analyze the top 5 tickers
        print(f"Analyzing {symbol}...")
        data = get_stock_data(symbol, interval='1h', days=30)

        if data.empty:
            print(f"No data found for {symbol}. Skipping...\n")
            continue

        # Calculate Indicators
        data = calculate_moving_averages(data)
        data = calculate_rsi(data)
        data = calculate_atr(data)

        # Generate Recommendations
        result = provide_trading_levels(data, symbol)

        if result["recommendation"] in ["Buy Now", "Strong Buy - Oversold"]:
            atr = data['ATR'].iloc[-1]
            potential_profit = atr * 2 / result['close_price'] * 100
            if potential_profit >= 50:  # At least 50% potential profit
                recommendations.append(result)

    # Sort recommendations by potential profit (descending)
    recommendations = sorted(recommendations, key=lambda x: x['target_profit'], reverse=True)

    print("\nTop Recommendations:")
    for rec in recommendations:
        print(f"Symbol: {rec['symbol']}")
        print(f"Current Price: ${rec['close_price']:.2f}")
        print(f"Stop Loss: ${rec['stop_loss']:.2f}")
        print(f"Target Profit: ${rec['target_profit']:.2f}")
        print(f"Support Level: ${rec['support']:.2f}, Resistance Level: ${rec['resistance']:.2f}")
        print(f"Recommendation: {rec['recommendation']}\n")

    # Send SMS with recommendations
    if recommendations:
        send_sms(recommendations)

if __name__ == "__main__":
    main()
