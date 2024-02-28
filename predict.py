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

# Assuming fetch_data, add_technical_indicators, and define_target_variable are defined in their respective modules
from utils.data_fetcher import fetch_data
from utils.feature_engineering import add_technical_indicators, define_target_variable

def predict_next_day_movement(stock_symbol, model_path):
    # Fetch the most recent data
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')  # Current date
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')  # One year ago
    data = fetch_data(stock_symbol, start_date, end_date)

    # Apply feature engineering
    indicators = [
        # {'name': 'SMA_10', 'type': 'SMA', 'window': 10},
        {'name': 'SMA_30', 'type': 'SMA', 'window': 30},
        # Add other indicators as needed
    ]
    data = add_technical_indicators(data, indicators)
    data = define_target_variable(data, 'target', 1)  # Assuming 'target' is defined as next-day movement

    # Load the trained model
    model = joblib.load(model_path)

    # Prepare the feature vector for the latest data point
    latest_features = data.iloc[-1][[indicator['name'] for indicator in indicators]].fillna(method='ffill').to_numpy().reshape(1, -1)

    # Predict the movement
    prediction = model.predict(latest_features)
    return prediction

if __name__ == "__main__":
    stock_symbol = 'NVDA'
    model_path = 'C:/Users/User/Desktop/algorithmic-trading-system/best_random_forest_model_epoch_1_accuracy_0.5827.joblib'  # Update this path
    prediction = predict_next_day_movement(stock_symbol, model_path)
    print(f"Predicted movement for {stock_symbol} tomorrow: {'Up' if prediction[0] == 1 else 'Down'}")
