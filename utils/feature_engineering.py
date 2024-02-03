
from data_fetcher import fetch_data

def calculate_moving_average(data, column, window, ma_type='SMA'):
    if ma_type == 'SMA':
        return data[column].rolling(window=window).mean()
    elif ma_type == 'EMA':
        return data[column].ewm(span=window, adjust=False).mean()

def calculate_rsi(data, column, window):
    delta = data[column].diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = -delta.clip(upper=0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, column, short_window, long_window, signal_window):
    short_ema = calculate_moving_average(data, column, short_window, 'EMA')
    long_ema = calculate_moving_average(data, column, long_window, 'EMA')
    data['MACD'] = short_ema - long_ema
    data['MACD_Signal'] = calculate_moving_average(data, 'MACD', signal_window, 'EMA')
    data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
    return data

def add_technical_indicators(data, indicators):
    for indicator in indicators:
        if indicator['type'] in ['SMA', 'EMA']:
            data[indicator['name']] = calculate_moving_average(data, 'Close', indicator['window'], indicator['type'])
        elif indicator['type'] == 'RSI':
            data[indicator['name']] = calculate_rsi(data, 'Close', indicator['window'])
        elif indicator['type'] == 'MACD':
            data = calculate_macd(data, 'Close', indicator['short_window'], indicator['long_window'], indicator['signal_window'])
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
        {'name': 'EMA_10', 'type': 'EMA', 'window': 10},
        {'name': 'RSI_14', 'type': 'RSI', 'window': 14},
        {'type': 'MACD', 'short_window': 12, 'long_window': 26, 'signal_window': 9}
    ]

    enhanced_data = add_technical_indicators(data, indicators)
    enhanced_data = define_target_variable(enhanced_data, 'target', 1)
    print(enhanced_data.head())

