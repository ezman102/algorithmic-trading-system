#backtester.py
import sys
import os
script_dir = os.path.dirname(__file__) 
parent_dir = os.path.dirname(script_dir) 
sys.path.append(parent_dir)

class Backtester:
    def __init__(self, data, model):
        self.data = data
        self.model = model

    def simulate_trading(self):
        features = self.data.drop('target', axis=1)
        target = self.data['target']
        predictions = self.model.predict(features)
        profit_per_trade = 10
        loss_per_trade = -10
        correct_predictions = sum(1 for pred, actual in zip(predictions, target) if pred == actual)
        incorrect_predictions = sum(1 for pred, actual in zip(predictions, target) if pred != actual)
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Incorrect Predictions: {incorrect_predictions}")

        cumulative_profit_loss = []
        current_profit_loss = 0

        for pred, actual in zip(predictions, target):
            if pred == actual:
                current_profit_loss += profit_per_trade
            else:
                current_profit_loss += loss_per_trade
            cumulative_profit_loss.append(current_profit_loss)

        return cumulative_profit_loss