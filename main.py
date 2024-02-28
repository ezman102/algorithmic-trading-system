#main.py
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, 'utils')
sys.path.append(utils_dir)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.impute import SimpleImputer
import joblib
from utils.data_fetcher import fetch_data
from utils.feature_engineering import add_technical_indicators, define_target_variable

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from utils.visualization import visualize_decision_trees, visualize_classification_results, visualize_regression_results

def preprocess_data(features):
    imputer = SimpleImputer(strategy='mean')  # Or any other strategy you prefer
    return imputer.fit_transform(features)

def main():
    stock_symbol = 'NVDA'
    start_date = '2023-01-01'
    end_date = '2024-02-28'

    print("Fetching data...")
    data = fetch_data(stock_symbol, start_date, end_date)
    data = add_technical_indicators(data, [
        # {'name': 'SMA_10', 'type': 'SMA', 'window': 10},
        # {'name': 'SMA_30', 'type': 'SMA', 'window': 30},
        # {'name': 'EMA_10', 'type': 'EMA', 'window': 10},
        # {'name': 'EMA_30', 'type': 'EMA', 'window': 30},
        # {'name': 'RSI_14', 'type': 'RSI', 'window': 14},
        # {'name': 'MACD', 'type': 'MACD', 'short_window': 12, 'long_window': 26, 'signal_window': 9},
        # {'name': 'MACD_Signal', 'type': 'MACD', 'short_window': 12, 'long_window': 26, 'signal_window': 9},
        {'name': 'MACD_Histogram', 'type': 'MACD', 'short_window': 12, 'long_window': 26, 'signal_window': 9},
        # {'name': 'BB_Upper', 'type': 'BB', 'window': 20},  # Bollinger Bands
        # {'name': 'BB_Lower', 'type': 'BB', 'window': 20},  # Bollinger Bands
        {'name': 'ATR', 'type': 'ATR', 'window': 14},  # Average True Range
        # {'name': 'Stochastic_Oscillator', 'type': 'Stochastic', 'window': 14},
        # {'name': 'OBV', 'type': 'OBV'}  # On-Balance Volume doesn't need a window
    ],drop_original=True)
    original_data = data.copy()
    print("Select mode:")
    print("1. Classification")
    print("2. Regression")
    choice = input("Enter choice (1/2): ")
    
    if choice == '1':
        mode = 'classification'
    elif choice == '2':
        mode = 'regression'
    else:
        print("Invalid choice. Exiting...")
        sys.exit(1)

    if mode == 'classification':
        model = RandomForestClassifier(random_state=42)
        data = define_target_variable(data, 'target_class', 1)  # for binary classification
        target_column = 'target_class'
    elif mode == 'regression':
        model = RandomForestRegressor(random_state=42)
        data = define_target_variable(data, 'target_reg', 0, is_regression=True)  # for continuous target
        target_column = 'target_reg'
    else:
        print("Invalid mode selected. Exiting...")
        return
    
    # data.drop(columns=['Close'], inplace=True, errors='ignore')

    features = data.drop([target_column], axis=1)
    target = data[target_column]

    # Preprocess the data to fill NaN values
    preprocessed_features = preprocess_data(features)

    X_train, X_test, y_train, y_test = train_test_split(preprocessed_features, target, test_size=0.2, random_state=42)
    

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    predictions = best_model.predict(X_test)
    latest_data = original_data.iloc[-1:][features.columns].fillna(method='ffill').fillna(method='bfill').values
    if mode == 'classification':
        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        print(f"Classification Accuracy: {accuracy}")
        print("Classification Report:\n", classification_report(y_test, predictions))
        print("Features used for training:", data.columns)
        next_day_prediction = best_model.predict(latest_data)
        print(f"Predicted class for the next day: {next_day_prediction[0]}")
        # Save the model with accuracy in the filename
        model_filename = f'best_random_forest_model_accuracy_{accuracy:.4f}.joblib'
        joblib.dump(best_model, model_filename)
        print(f"Model saved as {model_filename}")
        visualize_classification_results(y_test, predictions)
        visualize_decision_trees(best_model, features.columns, max_trees=1)


    elif mode == 'regression':
        # Calculate mean squared error
        mse = mean_squared_error(y_test, predictions)
        print(f"Regression MSE: {mse}")
        next_day_prediction = best_model.predict(latest_data)
        print(f"Predicted value for the next day: {next_day_prediction[0]}")
        
        # Save the model without accuracy since it's a regression model
        model_filename = 'best_random_forest_regression_model.joblib'
        joblib.dump(best_model, model_filename)
        print(f"Model saved as {model_filename}")
        visualize_regression_results(y_test, predictions)
        visualize_decision_trees(best_model, features.columns, max_trees=1)


if __name__ == "__main__":
    main()