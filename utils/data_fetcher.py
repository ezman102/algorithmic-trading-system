# data_fetcher.py
import yfinance as yf
import pandas as pd

def fetch_data(stock_symbol, start_date, end_date):
    """
    Fetch historical data for a given stock symbol from Yahoo Finance.
    """
    try:
        # Download stock data from Yahoo Finance
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
        # Check if data was successfully fetched
        if stock_data.empty:
            raise ValueError("No data fetched for the given symbol and date range.")
        print("Fetched data:")
        print(stock_data)
        return stock_data

    except Exception as e:
        print(f"An error occurred while fetching data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":

    default_stock_symbol = "AAPL"
    default_start_date = "2023-01-01"
    default_end_date = "2023-12-31"

    stock_symbol = input(f"Enter stock symbol (default is {default_stock_symbol}): ") or default_stock_symbol
    start_date = input(f"Enter start date (YYYY-MM-DD, default is {default_start_date}): ") or default_start_date
    end_date = input(f"Enter end date (YYYY-MM-DD, default is {default_end_date}): ") or default_end_date

    fetched_data = fetch_data(stock_symbol, start_date, end_date)
    