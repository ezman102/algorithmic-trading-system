#evaluate_combinations.py

import sys
import os
import pandas as pd
from itertools import combinations
from joblib import Parallel, delayed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.classification_model import ClassificationModel
from utils.backtester import Backtester
from utils.data_fetcher import fetch_data
from utils.feature_engineering import add_technical_indicators

def evaluate_combination(index, subset, data):
    print(f"Evaluating combination {index + 1}: {subset}")
    features_subset = data[list(subset)].fillna(data.mean())
    target = data['target']

    split_index = int(len(features_subset) * 0.9)
    X_train, X_test = features_subset.iloc[:split_index], features_subset.iloc[split_index:]
    y_train, y_test = target.iloc[:split_index], target.iloc[split_index:]

    model = ClassificationModel(n_estimators=300, max_depth=None, random_state=42)
    trained_model = model.train(X_train, y_train)

    test_data = pd.concat([X_test, y_test], axis=1)
    backtester = Backtester(test_data, trained_model)
    net_profit = backtester.simulate_trading()
    
    print(f"Combination {index + 1} Net Profit: {net_profit}")
    
    return (subset, net_profit)


def evaluate_feature_combinations_parallel(data, all_features, max_features=5):
    all_combinations = [comb for r in range(1, max_features + 1) for comb in combinations(all_features, r)]
    
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_combination)(i, subset, data) for i, subset in enumerate(all_combinations)
    )

    best_combination, max_profit = max(results, key=lambda x: x[1])
    return best_combination, max_profit



def main():
    stock_symbol = 'MSFT'
    start_date = '2023-03-20'
    end_date = '2024-03-20'

    print("Fetching data...")
    data = fetch_data(stock_symbol, start_date, end_date)

    print("Adding technical indicators...")
    indicators = [
        {'name': 'SMA_10', 'type': 'SMA', 'window': 10},
        {'name': 'SMA_30', 'type': 'SMA', 'window': 30},
        {'name': 'EMA_10', 'type': 'EMA', 'window': 10},
        {'name': 'EMA_30', 'type': 'EMA', 'window': 30},
        {'name': 'RSI_14', 'type': 'RSI', 'window': 14},
        {'name': 'MACD', 'type': 'MACD', 'short_window': 12, 'long_window': 26, 'signal_window': 9},
        {'name': 'MACD_Signal', 'type': 'MACD', 'short_window': 12, 'long_window': 26, 'signal_window': 9},
        {'name': 'MACD_Histogram', 'type': 'MACD', 'short_window': 12, 'long_window': 26, 'signal_window': 9},
        {'name': 'BB_Upper', 'type': 'BB', 'window': 20},  # Bollinger Bands
        {'name': 'BB_Lower', 'type': 'BB', 'window': 20},  # Bollinger Bands
        {'name': 'ATR', 'type': 'ATR', 'window': 14},  # Average True Range
        {'name': 'Stochastic_Oscillator', 'type': 'Stochastic', 'window': 14},
        {'name': 'OBV', 'type': 'OBV'}  # On-Balance Volume doesn't need a window
    ]
    data = add_technical_indicators(data, indicators, drop_original=False)
    print(data)
    print("Defining target variable...")
    # Modify the definition according to your needs
    data['target'] = (data['Close'].shift(-1) > data['Close']).astype(int)  # Example for a binary classification target

    all_features = [indicator['name'] for indicator in indicators]
    best_combination = evaluate_feature_combinations_parallel(data, all_features)
    print(f"Best feature combination: {best_combination}")

if __name__ == "__main__":
    main()

