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

root = tk.Tk()
root.title("Trading System Backtester")

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

        messagebox.showinfo("Result", f"Simulated Profit/Loss: {profit_loss}")
    except Exception as e:
        messagebox.showerror("Error", str(e))



# Create and place the widgets
tk.Label(root, text="Stock Symbol").grid(row=0, column=0)
stock_symbol_entry = tk.Entry(root)
stock_symbol_entry.grid(row=0, column=1)

tk.Label(root, text="Start Date (YYYY-MM-DD)").grid(row=1, column=0)
start_date_entry = tk.Entry(root)
start_date_entry.grid(row=1, column=1)

tk.Label(root, text="End Date (YYYY-MM-DD)").grid(row=2, column=0)
end_date_entry = tk.Entry(root)
end_date_entry.grid(row=2, column=1)

# Create checkboxes for each indicator
row = 3
for indicator, var in indicator_selections.items():
    tk.Checkbutton(root, text=indicator, variable=var).grid(row=row, column=0, sticky='w')
    row += 1

run_button = tk.Button(root, text="Run Backtest", command=run_backtest)
run_button.grid(row=row, column=0, columnspan=2)

# Start the main loop
root.mainloop()
