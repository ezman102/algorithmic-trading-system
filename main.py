import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

utils_dir = os.path.join(script_dir, 'utils')
sys.path.append(utils_dir)

import pandas as pd
from utils.data_fetcher import fetch_data
from utils.feature_engineering import add_technical_indicators
from utils.feature_engineering import define_target_variable
from models.random_forest_model import RandomForestModel
from utils.backtester import Backtester
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils.visualize_decision_trees import visualize_decision_trees

def main():
    # Set your stock symbol and date range for historical data
    stock_symbol = 'AAPL'  # Example stock symbol
    start_date = '2020-01-01'
    end_date = '2023-01-01'

    # Step 1: Fetch Data
    print("Fetching data...")
    data = fetch_data(stock_symbol, start_date, end_date)

    # Define the indicators you want to calculate
    indicators = [
        {'name': 'SMA_10', 'type': 'SMA', 'window': 10},
        {'name': 'SMA_30', 'type': 'SMA', 'window': 30},
        {'name': 'EMA_10', 'type': 'EMA', 'window': 10},
        {'name': 'EMA_30', 'type': 'EMA', 'window': 30},
        {'name': 'RSI_14', 'type': 'RSI', 'window': 14},
        {'type': 'MACD', 'short_window': 12, 'long_window': 26, 'signal_window': 9} # Added here
    ]

    # Step 2: Feature Engineering
    print("Engineering features...")
    data = add_technical_indicators(data, indicators)

    # Ensure 'target' column is defined
    data = define_target_variable(data, 'target', 1)

    # Ask the user for the desired features
    print("Available features:", list(data.columns))
    selected_features = input("Enter the features you want to use, separated by commas (e.g., Open,High,Low,Close): ")
    selected_features = selected_features.split(',')

    # Verify selected features
    for feature in selected_features:
        if feature not in data.columns:
            print(f"Feature {feature} is not available. Exiting...")
            return

    # Step 3: Preparing Data for Training
    print("Preparing data for training...")
    features = data[selected_features]
    target = data['target']

    # Handle NaN values in selected features
    features.fillna(features.mean(), inplace=True)  # Fill NaN with column mean or choose another method

    data.to_csv('data/enhanced_stock_data.csv', index=False)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Step 4: Model Training
    print("Training model...")
    model = RandomForestModel(n_estimators=100, random_state=42)
    # model = RandomForestModel(n_estimators=100, max_depth=5, random_state=42)
    model.train(X_train, y_train)

    # Step 5: Backtesting
    print("Backtesting...")
    backtester = Backtester(pd.concat([X_test, y_test], axis=1), model)
    profit_loss = backtester.simulate_trading()
    print(f"Simulated Profit/Loss: {profit_loss}")

    visualize_decision_trees(model.model, features.columns, max_trees=3)

    # num_trees_to_visualize = 3  
    # for i in range(min(num_trees_to_visualize, len(model.model.estimators_))):
    #     tree = model.model.estimators_[i]
    #     visualize_decision_tree(tree, features.columns, ["Down", "Up"], "Stock Price Movement")

if __name__ == "__main__":
    main()
