#feature_engineering.py

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
    macd = short_ema - long_ema
    macd_signal = macd.ewm(span=signal_window, adjust=False).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

def add_technical_indicators(data, indicators, drop_original=True):
    for indicator in indicators:
        if indicator['type'] == 'SMA' or indicator['type'] == 'EMA':
            data[indicator['name']] = calculate_moving_average(data, 'Close', indicator['window'], indicator['type'])
        elif indicator['type'] == 'RSI':
            data[indicator['name']] = calculate_rsi(data, 'Close', indicator['window'])
        elif indicator['type'] == 'MACD':
            # Ensure this only runs if the indicator type is MACD and we are looking for MACD related calculations
            macd, macd_signal, macd_histogram = calculate_macd(data, 'Close', indicator['short_window'], indicator['long_window'], indicator['signal_window'])
            data['MACD'] = macd
            data['MACD_Signal'] = macd_signal
            data['MACD_Histogram'] = macd_histogram
    
    if drop_original:
        # Drop the original OHLCV columns
        data.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True, errors='ignore')
    
    return data

def define_target_variable(data, target_column, shift_period, is_regression=False):
    """
    Defines the target variable for the dataset.

    Parameters:
    - data: The DataFrame containing your data.
    - target_column: The name of the new target column to be created.
    - shift_period: The period by which to shift the 'Close' column to create the target.
    - is_regression: A boolean indicating whether the target variable is for regression (True) or classification (False).

    Returns:
    - The DataFrame with the new target variable added.
    """
    if is_regression:
        # For regression, predict the actual next day's 'Close' price or another continuous value
        data[target_column] = data['Close'].shift(-shift_period)
    else:
        # For classification, predict whether the next day's 'Close' price will be higher (1) or not (0)
        data[target_column] = (data['Close'].shift(-shift_period) > data['Close']).astype(int)

    # Drop rows with NaN values that result from shifting
    data.dropna(subset=[target_column], inplace=True)
    return data




