import sys
import os
import tkinter as tk
import pandas as pd
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from itertools import chain, combinations

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, 'utils')
sys.path.append(utils_dir)

from utils.data_fetcher import fetch_data
from utils.feature_engineering import add_technical_indicators, define_target_variable
from models.random_forest_model import RandomForestModel
from utils.backtester import Backtester
from utils.visualize_decision_trees import visualize_decision_trees

class TradingSystemBacktester:
    def __init__(self, root):
        self.root = root
        self.model = None
        self.features = None
        self.setup_ui()
    def setup_ui(self):
        self.stock_symbol_var, _ = self.create_entry_with_label("Stock Symbol", 0, "")
        self.start_date_var, _ = self.create_entry_with_label("Start Date", 1, "2020-01-01")
        self.end_date_var, _ = self.create_entry_with_label("End Date", 2, "2023-11-16")

        tk.Label(self.root, text="Market Data Features").grid(row=3, column=0)
        market_data_features = ['Open', 'Close', 'High', 'Low', 'Volume']
        self.market_data_selections = self.create_checkboxes(market_data_features, 4)

        tk.Label(self.root, text="Technical Indicators").grid(row=9, column=0)
        available_indicators = ['SMA_10', 'SMA_30', 'EMA_10', 'EMA_30', 'RSI_14', 'MACD']
        self.indicator_selections = self.create_checkboxes(available_indicators, 10)

        run_button = tk.Button(self.root, text="Run Backtest", command=self.run_backtest)
        run_button.grid(row=16, column=0, columnspan=2)

        find_best_button = tk.Button(self.root, text="Find Best Combination", command=self.find_best_combination)
        find_best_button.grid(row=17, column=0, columnspan=2, sticky='we')

        visualize_button = tk.Button(self.root, text="Visualize Decision Trees", command=self.visualize_trees)
        visualize_button.grid(row=18, column=0, columnspan=2, sticky='we')


    def create_entry_with_label(self, label_text, row, default_value=""):
        tk.Label(self.root, text=label_text).grid(row=row, column=0)
        var = tk.StringVar(value=default_value)
        entry = tk.Entry(self.root, textvariable=var)
        entry.grid(row=row, column=1)
        return var, entry

    def create_checkboxes(self, items, start_row):
        vars = {}
        for i, item in enumerate(items, start=start_row):
            var = tk.BooleanVar()
            tk.Checkbutton(self.root, text=item, variable=var).grid(row=i, column=0, sticky='w')
            vars[item] = var
        return vars

    def run_backtest(self):
        stock_symbol = self.stock_symbol_var.get()
        start_date = self.start_date_var.get()
        end_date = self.end_date_var.get()
        
        try:
            data = fetch_data(stock_symbol, start_date, end_date)
            
            selected_market_data = [feature for feature, var in self.market_data_selections.items() if var.get()]
            selected_indicators = [self.parse_indicator_selection(indicator, var.get()) for indicator, var in self.indicator_selections.items() if var.get()]
            
            # Add market data features directly to the selected features list
            selected_features = selected_market_data

            # Process technical indicators
            if selected_indicators:
                data = add_technical_indicators(data, selected_indicators)
                selected_features += [indicator['name'] for indicator in selected_indicators]

            data = define_target_variable(data, 'target', 1)

            # Filter the data for only selected features plus the target
            self.features = data[selected_features + ['target']]
            target = data['target']
            X_train, X_test, y_train, y_test = train_test_split(self.features, target, test_size=0.2, random_state=42)

            self.model = RandomForestModel(n_estimators=100, random_state=42)
            self.model.train(X_train, y_train)

            backtester = Backtester(pd.concat([X_test, y_test], axis=1), self.model)
            profit_loss = backtester.simulate_trading()

            balance = data['target'].value_counts()
            print("Balance of target variable:\n", balance)

            messagebox.showinfo("Result", f"Simulated Profit/Loss: {profit_loss}")
        except Exception as e:
            messagebox.showerror("Error", str(e))



    def find_best_combination(self):
        stock_symbol = self.stock_symbol_var.get()
        start_date = self.start_date_var.get()
        end_date = self.end_date_var.get()
        best_profit = float('-inf')
        best_combination = None

        try:
            data = fetch_data(stock_symbol, start_date, end_date)
            feature_names = ['SMA_10', 'SMA_30', 'EMA_10', 'EMA_30', 'RSI_14', 'MACD', 'Open', 'Close', 'High', 'Low', 'Volume']
            all_feature_combinations = list(self.all_subsets(feature_names))

            for combination in all_feature_combinations:
                selected_indicators = [self.parse_indicator_selection(feature, True) for feature in combination]
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

    def visualize_trees(self):
        if self.model and self.features is not None:
            try:
                visualize_decision_trees(self.model.model, self.features.columns, max_trees=3)
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while visualizing trees: {e}")
        else:
            messagebox.showinfo("Information", "Please run backtest first to train the model.")

    def parse_indicator_selection(self, indicator, selected):
        if indicator == 'MACD':
            return {'type': 'MACD', 'short_window': 12, 'long_window': 26, 'signal_window': 9}
        else:
            return {'name': indicator, 'type': indicator[:3], 'window': int(indicator.split('_')[-1])}

    def all_subsets(self, lst):
        return chain(*map(lambda x: combinations(lst, x), range(1, len(lst)+1)))

# Usage
root = tk.Tk()
app = TradingSystemBacktester(root)
root.mainloop()
