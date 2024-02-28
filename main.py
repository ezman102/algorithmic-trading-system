import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

utils_dir = os.path.join(script_dir, 'utils')
sys.path.append(utils_dir)

import pandas as pd
import numpy as np
import joblib
from utils.data_fetcher import fetch_data
from utils.feature_engineering import add_technical_indicators
from utils.feature_engineering import define_target_variable
from models.random_forest_model import RandomForestModel
from utils.backtester import Backtester
import matplotlib.pyplot as plt
from utils.visualization import visualize_decision_trees
from utils.evaluate_combinations import evaluate_feature_combinations_parallel
from sklearn.metrics import accuracy_score


def main():
    # Set your stock symbol and date range for historical data
    stock_symbol = 'nvda'  # Example stock symbol
    start_date = '2018-01-01'
    end_date = '2023-01-01'
    
    # start_date = '2019-01-01'
    # end_date = '2024-02-28'

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
    data = define_target_variable(data, 'target', 1)

    # Ask the user for the desired operation mode
    mode = input("Enter 'manual' to select features manually or 'auto' for automatic feature combination evaluation: ")
    

    if mode == 'manual':
        print("Available features:", list(data.columns))
        selected_features = input("Enter the features you want to use, separated by commas: ")
        selected_features = selected_features.split(',')

        # Verify selected features
        for feature in selected_features:
            if feature not in data.columns:
                print(f"Feature {feature} is not available. Exiting...")
                return

        features = data[selected_features]
        features.fillna(features.mean(), inplace=True)

    elif mode == 'auto':
        print("Evaluating feature combinations (parallel processing)...")
        all_features = [col for col in data.columns if col != 'target']
        best_combination, max_profit = evaluate_feature_combinations_parallel(data, all_features)
        print(f"Best feature combination: {best_combination}")
        print(f"Maximum Profit/Loss: {max_profit}")
        selected_features = list(best_combination)

    else:
        print("Invalid mode selected. Exiting...")
        return

    # Step 3: Preparing Data for Training
    print("Preparing data for training...")

    features = data[selected_features]
    target = data['target']  

    # Handle NaN values in selected features
    features.fillna(features.mean(), inplace=True)

    # Calculate the index for splitting the data
    split_index = int(len(data) * 0.9)  # 80% for training, 20% for testing

    # Split the data in a time-ordered manner
    X_train, X_test = features.iloc[:split_index], features.iloc[split_index:]
    y_train, y_test = target.iloc[:split_index], target.iloc[split_index:]

    # Step 4: Model Training with 'epochs' and variable random seeds
    print("Training model for 50 epochs...")
    best_accuracy = 0
    best_epoch = 0
    for epoch in range(50):
        # Change the random seed in each iteration
        random_seed = np.random.randint(0, 10000)  # Generates a new random seed
        model = RandomForestModel(n_estimators=200, max_depth=20, min_samples_leaf=2, min_samples_split=5, random_state=random_seed)
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Epoch {epoch+1}, Random Seed: {random_seed}: Accuracy: {accuracy}")
        
        # Optionally, save the model if it has the best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            best_model = model
            
            # Save the model to disk
            model_filename = f'best_random_forest_model_epoch_{epoch+1}_accuracy_{best_accuracy:.4f}.joblib'
            joblib.dump(best_model, os.path.join(script_dir, model_filename))
            print(f"Model saved to {model_filename}")           

            # loaded_model = joblib.load('path_to_your_saved_model.joblib')
            
    print(f"Best Accuracy: {best_accuracy} on Epoch: {best_epoch+1} with Random Seed: {random_seed}")

    # Step 5: Backtesting
    print("Backtesting...")
    backtester = Backtester(pd.concat([X_test, y_test], axis=1), model)
    profit_loss = backtester.simulate_trading()
    print(f"Simulated Profit/Loss: {profit_loss}")

    visualize_decision_trees(model.model, features.columns, max_trees=1)

    
    # Run the backtest simulation
    cumulative_profit_loss = backtester.simulate_trading()

    # Convert the cumulative profit/loss to a pandas Series
    profit_loss_series = pd.Series(cumulative_profit_loss, index=X_test.index)

    # Plot the strategy returns as a line chart
    plt.figure(figsize=(12, 6))
    plt.plot(profit_loss_series, label='Strategy Cumulative Returns')
    plt.title('Cumulative Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
