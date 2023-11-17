import sys
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
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

def create_entry_with_label(root, label_text, row, default_value=""):
    tk.Label(root, text=label_text).grid(row=row, column=0)
    var = tk.StringVar(value=default_value)
    entry = tk.Entry(root, textvariable=var)
    entry.grid(row=row, column=1)
    return var, entry

def create_checkboxes(root, items, start_row):
    vars = {}
    for i, item in enumerate(items, start=start_row):
        var = tk.BooleanVar()
        tk.Checkbutton(root, text=item, variable=var).grid(row=i, column=0, sticky='w')
        vars[item] = var
    return vars

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


global stock_symbol_entry, start_date_entry, end_date_entry

def create_entry_with_label(root, label_text, row, default_value=""):
    tk.Label(root, text=label_text).grid(row=row, column=0)
    var = tk.StringVar(value=default_value)
    entry = tk.Entry(root, textvariable=var)
    entry.grid(row=row, column=1)
    return var, entry

root = tk.Tk()
root.title("Trading System Backtester")

default_values = {"Stock Symbol": "", "Start Date": "2020-01-01", "End Date": "2023-11-16"}
stock_symbol_var, stock_symbol_entry = create_entry_with_label(root, "Stock Symbol", 0, default_values["Stock Symbol"])
start_date_var, start_date_entry = create_entry_with_label(root, "Start Date", 1, default_values["Start Date"])
end_date_var, end_date_entry = create_entry_with_label(root, "End Date", 2, default_values["End Date"])

# Define default values
stock_symbol_var, stock_symbol_entry = create_entry_with_label(root, "Stock Symbol", 0, default_values["Stock Symbol"])

available_indicators = ['SMA_10', 'SMA_30', 'EMA_10', 'EMA_30', 'RSI_14', 'MACD']
indicator_selections = create_checkboxes(root, available_indicators, len(default_values) + 1)

run_button = tk.Button(root, text="Run Backtest", command=run_backtest)
run_button.grid(row=len(default_values) + len(available_indicators) + 1, column=0, columnspan=2)

find_best_button = tk.Button(root, text="Find Best Combination", command=find_best_combination)
find_best_button.grid(row=len(default_values) + len(available_indicators) + 2, column=0, columnspan=2, sticky='we')

root.mainloop()