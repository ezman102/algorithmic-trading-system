import pandas as pd
import numpy as np
from data_fetcher import fetch_data


def calculate_sma(data, column, window):
    return data[column].rolling(window=window).mean()

def calculate_ema(data, column, window):
    return data[column].ewm(span=window, adjust=False).mean()

def calculate_rsi(data, column, window):
    delta = data[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, column, short_window, long_window, signal_window):
    # Calculate the MACD Line
    short_ema = calculate_ema(data, column, short_window)
    long_ema = calculate_ema(data, column, long_window)
    data['MACD'] = short_ema - long_ema

    # Calculate the MACD Signal Line
    data['MACD_Signal'] = calculate_ema(data, 'MACD', signal_window)
    
    # Calculate the MACD Histogram
    data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
    return data

def add_technical_indicators(data, indicators):
    for indicator in indicators:
        if indicator['type'] == 'SMA':
            data[indicator['name']] = calculate_sma(data, 'Close', indicator['window'])
        elif indicator['type'] == 'EMA':
            data[indicator['name']] = calculate_ema(data, 'Close', indicator['window'])
        elif indicator['type'] == 'RSI':
            data[indicator['name']] = calculate_rsi(data, 'Close', indicator['window'])

    if any(indicator['type'] == 'MACD' for indicator in indicators):
        # Extract MACD settings from indicators list
        macd_settings = next(item for item in indicators if item["type"] == "MACD")
        data = calculate_macd(data, 'Close', 
                              macd_settings['short_window'],
                              macd_settings['long_window'],
                              macd_settings['signal_window'])
    return data


def define_target_variable(data, target_column, shift_period):
    data[target_column] = (data['Close'].shift(-shift_period) > data['Close']).astype(int)
    return data.dropna()

# Example usage
if __name__ == "__main__":
    stock_symbol = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    data = fetch_data(stock_symbol, start_date, end_date)

    indicators = [
        {'name': 'SMA_10', 'type': 'SMA', 'window': 10},
        {'name': 'SMA_30', 'type': 'SMA', 'window': 30},
        {'name': 'EMA_10', 'type': 'EMA', 'window': 10},
        {'name': 'EMA_30', 'type': 'EMA', 'window': 30},
        {'name': 'RSI_14', 'type': 'RSI', 'window': 14}
    ]
    
    indicators.append(
        {'type': 'MACD', 'short_window': 12, 'long_window': 26, 'signal_window': 9}
    )

    enhanced_data = add_technical_indicators(data, indicators)
    enhanced_data = define_target_variable(enhanced_data, 'target', 1)
    print(enhanced_data.head())
