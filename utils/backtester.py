import numpy as np

class Backtester:
    def __init__(self, data, model, initial_capital=10000, trade_size=1000, stop_loss_percent=0.05, take_profit_percent=0.10, trading_fee=0.001):
        self.data = data
        self.model = model
        self.initial_capital = initial_capital
        self.trade_size = trade_size
        self.stop_loss_percent = stop_loss_percent
        self.take_profit_percent = take_profit_percent
        self.trading_fee = trading_fee

    def simulate_trading(self):
        capital = self.initial_capital
        features = self.data.drop('target', axis=1)
        target = self.data['target']
        predictions = self.model.predict(features)
        
        for pred, actual in zip(predictions, target):
            if pred == actual:  # Correct prediction
                profit_or_loss = self.trade_size * self.take_profit_percent
            else:  # Incorrect prediction
                profit_or_loss = -self.trade_size * self.stop_loss_percent
            
            # Subtract trading fee from profit or loss
            profit_or_loss -= self.trade_size * self.trading_fee

            capital += profit_or_loss

            # Stop trading if capital falls below a certain threshold to simulate a margin call or risk management stop
            if capital <= self.initial_capital * 0.5:
                print("Stopping trading due to low capital.")
                break

        net_profit = capital - self.initial_capital
        return net_profit
