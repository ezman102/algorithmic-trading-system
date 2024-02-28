import sys
import os

# Add the parent directory to sys.path
script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
parent_dir = os.path.dirname(script_dir)  # Get the parent directory
sys.path.append(parent_dir)

import pandas as pd
from models.classification_model import ClassificationModel

class Backtester:
    def __init__(self, data, model):
        """
        Initialize the Backtester.

        Parameters:
        data (pandas.DataFrame): The dataset used for backtesting.
        model (RandomForestModel): The trained model for predictions.
        """
        self.data = data
        self.model = model

    def simulate_trading(self):
        """
        Simulate trading based on the model's predictions.
        Returns:
        float: The simulated profit/loss of the strategy.
        """
        # Split data into features and target
        features = self.data.drop('target', axis=1)
        target = self.data['target']
        predictions = self.model.predict(features)
        # Trading strategy here
        # For example, buy (1) if prediction is positive, hold (0) otherwise
        # Calculate profit/loss based on the strategy
        # For simplicity, Assume each correct prediction yields a fixed profit, each wrong prediction yields a fixed loss
        profit_per_trade = 10
        loss_per_trade = -10
        correct_predictions = sum(1 for pred, actual in zip(predictions, target) if pred == actual)
        incorrect_predictions = sum(1 for pred, actual in zip(predictions, target) if pred != actual)
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Incorrect Predictions: {incorrect_predictions}")

        # Instead of printing, store the profit/loss in a list
        cumulative_profit_loss = []
        current_profit_loss = 0

        for pred, actual in zip(predictions, target):
            if pred == actual:
                current_profit_loss += profit_per_trade
            else:
                current_profit_loss += loss_per_trade
            cumulative_profit_loss.append(current_profit_loss)

        # Return the time series of cumulative profit/loss
        return cumulative_profit_loss
        # return profit_loss

