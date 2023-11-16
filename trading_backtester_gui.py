import sys
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the utils directory to sys.path
utils_dir = os.path.join(script_dir, 'utils')
sys.path.append(utils_dir)

import tkinter as tk
import pandas as pd
from tkinter import messagebox
from utils.data_fetcher import fetch_data
from utils.feature_engineering import add_technical_indicators, define_target_variable
from models.random_forest_model import RandomForestModel
from utils.backtester import Backtester
from sklearn.model_selection import train_test_split
from itertools import chain, combinations


def all_subsets(lst):
    """Return non-empty subsets of lst."""
    return chain(*map(lambda x: combinations(lst, x), range(1, len(lst)+1)))

root = tk.Tk()
root.title("Trading System Backtester")

default_stock_symbol = "AAPL"
default_start_date = "2020-01-01"
default_end_date = "2023-11-16"

# Define StringVar for entry fields with default values
stock_symbol_var = tk.StringVar(value=default_stock_symbol)
start_date_var = tk.StringVar(value=default_start_date)
end_date_var = tk.StringVar(value=default_end_date)

# Define available indicators
available_indicators = [
    'SMA_10', 'SMA_30', 'EMA_10', 'EMA_30', 'RSI_14', 'MACD'
]

# Dictionary to track checkbox states
indicator_selections = {indicator: tk.BooleanVar() for indicator in available_indicators}

def run_backtest():
    stock_symbol = stock_symbol_entry.get()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()
    
    try:
        # Fetch data
        data = fetch_data(stock_symbol, start_date, end_date)

        # List of selected indicators
        selected_indicators = []
        for indicator, var in indicator_selections.items():
            if var.get():
                if indicator == 'MACD':
                    selected_indicators.append({'type': 'MACD', 'short_window': 12, 'long_window': 26, 'signal_window': 9})
                else:
                    selected_indicators.append({'name': indicator, 'type': indicator[:3], 'window': int(indicator.split('_')[-1])})

        # Add features
        data = add_technical_indicators(data, selected_indicators)

        # Define target
        data = define_target_variable(data, 'target', 1)
        
        # Prepare data
        features = data.drop(['target'], axis=1)
        target = data['target']
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestModel(n_estimators=100, random_state=42)
        model.train(X_train, y_train)

        # Backtest
        backtester = Backtester(pd.concat([X_test, y_test], axis=1), model)
        profit_loss = backtester.simulate_trading()

        # Check the balance of the target variable
        balance = data['target'].value_counts()
        print("Balance of target variable:\n", balance)

        messagebox.showinfo("Result", f"Simulated Profit/Loss: {profit_loss}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def find_best_combination():
    stock_symbol = stock_symbol_var.get()
    start_date = start_date_var.get()
    end_date = end_date_var.get()
    best_profit = float('-inf')
    best_combination = None

    try:
        data = fetch_data(stock_symbol, start_date, end_date)
        feature_names = ['SMA_10', 'SMA_30', 'EMA_10', 'EMA_30', 'RSI_14', 'MACD']
        all_feature_combinations = list(all_subsets(feature_names))

        for combination in all_feature_combinations:
            selected_indicators = [{'name': feature, 'type': feature[:3], 'window': int(feature.split('_')[-1])}
                                   if 'MACD' not in feature
                                   else {'type': 'MACD', 'short_window': 12, 'long_window': 26, 'signal_window': 9}
                                   for feature in combination]

            data_with_indicators = add_technical_indicators(data.copy(), selected_indicators)
            data_with_target = define_target_variable(data_with_indicators, 'target', 1)
            features = data_with_target.drop(['target'], axis=1)
            target = data_with_target['target']

            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
            model = RandomForestModel(n_estimators=100, random_state=42)
            model.train(X_train, y_train)

            backtester = Backtester(pd.concat([X_test, y_test], axis=1), model)
            profit_loss = backtester.simulate_trading()

            if profit_loss > best_profit:
                best_profit = profit_loss
                best_combination = combination

        messagebox.showinfo("Best Combination Result", f"Best combination: {best_combination}\nProfit/Loss: {best_profit}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Create and place the widgets for entries
tk.Label(root, text="Stock Symbol").grid(row=0, column=0)
stock_symbol_entry = tk.Entry(root, textvariable=stock_symbol_var)
stock_symbol_entry.grid(row=0, column=1)

tk.Label(root, text="Start Date (YYYY-MM-DD)").grid(row=1, column=0)
start_date_entry = tk.Entry(root, textvariable=start_date_var)
start_date_entry.grid(row=1, column=1)

tk.Label(root, text="End Date (YYYY-MM-DD)").grid(row=2, column=0)
end_date_entry = tk.Entry(root, textvariable=end_date_var)
end_date_entry.grid(row=2, column=1)

# Create checkboxes for each indicator
row = 3
for indicator, var in indicator_selections.items():
    tk.Checkbutton(root, text=indicator, variable=var).grid(row=row, column=0, sticky='w')
    row += 1

run_button = tk.Button(root, text="Run Backtest", command=run_backtest)
run_button.grid(row=row, column=0, columnspan=2)

row += 1

find_best_button = tk.Button(root, text="Find Best Combination", command=find_best_combination)
find_best_button.grid(row=row, column=0, columnspan=2, sticky='we')

# Start the main loop
root.mainloop()
