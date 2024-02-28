import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

utils_dir = os.path.join(script_dir, 'utils')
sys.path.append(utils_dir)
# Assuming these are your custom modules, adjust the import paths as necessary
from utils.data_fetcher import fetch_data
from utils.feature_engineering import add_technical_indicators, define_target_variable
from models.classification_model import RandomForestModel
from utils.backtester import Backtester
from utils.visualization import visualize_decision_trees
from utils.evaluate_combinations import evaluate_feature_combinations_parallel

def main():
    # Fetch and preprocess data
    stock_symbol = 'NVDA'
    start_date = '2017-01-01'
    end_date = '2023-01-01'
    data = fetch_data(stock_symbol, start_date, end_date)
    indicators = [
        {'name': 'SMA_10', 'type': 'SMA', 'window': 10},
        {'name': 'SMA_30', 'type': 'SMA', 'window': 30},
        {'name': 'EMA_10', 'type': 'EMA', 'window': 10},
        {'name': 'EMA_30', 'type': 'EMA', 'window': 30},
        {'name': 'RSI_14', 'type': 'RSI', 'window': 14},
        {'type': 'MACD', 'short_window': 12, 'long_window': 26, 'signal_window': 9}

    ]
    data = add_technical_indicators(data, indicators)
    data = define_target_variable(data, 'target', 1)

    # Split the data
    features = data.drop('target', axis=1)
    target = data['target']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Perform GridSearchCV to find the best RandomForest parameters
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_)

    # Train a RandomForest model with the best parameters
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    print("Accuracy on test set:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Additional custom RandomForestModel training for epochs with variable seeds (example purpose)
    best_accuracy, best_epoch, best_model = 0, 0, None
    for epoch in range(50):
        random_seed = np.random.randint(0, 10000)
        model = RandomForestModel(n_estimators=500, random_state=random_seed)  # Assuming this is a wrapper around sklearn's RF
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            best_model = model
            model_filename = f'best_random_forest_model_epoch_{epoch+1}_accuracy_{best_accuracy:.4f}.joblib'
            joblib.dump(best_model, model_filename)
            print(f"Model saved to {model_filename}")
    print(f"Best Accuracy: {best_accuracy} on Epoch: {best_epoch+1}")

    # Simulate trading with backtesting (assuming you have this functionality)
    backtester = Backtester(pd.concat([X_test, y_test], axis=1), best_model)
    profit_loss = backtester.simulate_trading()
    print(f"Simulated Profit/Loss: {profit_loss}")

if __name__ == "__main__":
    main()
