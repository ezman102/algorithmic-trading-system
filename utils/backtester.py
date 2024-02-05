import sys
import os

# Add the parent directory to sys.path
script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
parent_dir = os.path.dirname(script_dir)  # Get the parent directory
sys.path.append(parent_dir)

import pandas as pd
from models.random_forest_model import RandomForestModel
from sklearn.model_selection import train_test_split

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
        profit_loss = sum(profit_per_trade if pred == actual else loss_per_trade 
                          for pred, actual in zip(predictions, target))
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

# Example usage
if __name__ == "__main__":
    # Load the data
    data = pd.read_csv("data/enhanced_stock_data.csv")

    # Print column names for debugging
    print("Column names in the dataset:", data.columns)

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestModel()
    model.train(X_train, y_train)

    # Initialize Backtester with the test data and trained model
    backtester = Backtester(X_test.join(y_test), model)
    
    # Run the simulation
    profit_loss = backtester.simulate_trading()
    print(f"Simulated Profit/Loss: {profit_loss}")
