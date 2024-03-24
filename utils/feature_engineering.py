#feature_engineering.py

import numpy as np
import pandas as pd

def add_rsi_threshold_features(data, rsi_column, thresholds):
    for threshold in thresholds:
        feature_name = f'RSI_above_{threshold}'
        data[feature_name] = (data[rsi_column] > threshold).astype(int)
    return data



def calculate_bollinger_bands(data, column, window=20, bands=['upper', 'lower']):
    rolling_mean = data[column].rolling(window=window).mean()
    rolling_std = data[column].rolling(window=window).std()
    if 'upper' in bands:
        upper_band = rolling_mean + (rolling_std * 2)
        data['BB_Upper'] = upper_band
    if 'lower' in bands:
        lower_band = rolling_mean - (rolling_std * 2)
        data['BB_Lower'] = lower_band


def calculate_atr(data, window=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(window=window).mean()
    data['ATR'] = atr

def calculate_stochastic_oscillator(data, window=14):
    low_min = data['Low'].rolling(window=window).min()
    high_max = data['High'].rolling(window=window).max()
    data['Stochastic_Oscillator'] = ((data['Close'] - low_min) / (high_max - low_min)) * 100

def calculate_obv(data):
    obv = [0]
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
            obv.append(obv[-1] + data['Volume'].iloc[i])
        elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
            obv.append(obv[-1] - data['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    data['OBV'] = obv

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

def add_technical_indicators(data, indicators, drop_original=False):
    macd_added = False  

    for indicator in indicators:
        if indicator['type'] == 'SMA' or indicator['type'] == 'EMA':
            data[indicator['name']] = calculate_moving_average(data, 'Close', indicator['window'], indicator['type'])
        elif indicator['type'] == 'RSI':
            data[indicator['name']] = calculate_rsi(data, 'Close', indicator['window'])
        elif indicator['type'] == 'MACD' and not macd_added:
            # Ensure MACD and its components are calculated only once
            macd, macd_signal, macd_histogram = calculate_macd(data, 'Close', indicator['short_window'], indicator['long_window'], indicator['signal_window'])
            if any(ind['name'] == 'MACD' for ind in indicators):
                data['MACD'] = macd
            if any(ind['name'] == 'MACD_Signal' for ind in indicators):
                data['MACD_Signal'] = macd_signal
            if any(ind['name'] == 'MACD_Histogram' for ind in indicators):
                data['MACD_Histogram'] = macd_histogram
            macd_added = True  # Mark MACD as added
        elif indicator['type'] == 'BB':
            bands_to_add = []
            if 'BB_Upper' in indicator['name']:
                bands_to_add.append('upper')
            if 'BB_Lower' in indicator['name']:
                bands_to_add.append('lower')
            calculate_bollinger_bands(data, 'Close', indicator['window'], bands=bands_to_add)
        elif indicator['type'] == 'ATR':
            calculate_atr(data, indicator['window'])
        elif indicator['type'] == 'Stochastic':
            calculate_stochastic_oscillator(data, indicator['window'])
        elif indicator['type'] == 'OBV':
            calculate_obv(data)
    
    if drop_original:
        data.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True, errors='ignore')
    
    return data


def define_target_variable(data, target_column, shift_period, is_regression=False):

    if is_regression:
        # For regression, predict the actual next day's 'Close' price or another continuous value
        data[target_column] = data['Close'].shift(-shift_period)
    else:
        # For classification, predict whether the next day's 'Close' price will be higher (1) or not (0)
        data[target_column] = (data['Close'].shift(-shift_period) > data['Close']).astype(int)

    # Drop rows with NaN values that result from shifting
    data.dropna(subset=[target_column], inplace=True)
    return data




