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

        return stock_data

    except Exception as e:
        print(f"An error occurred while fetching data: {e}")
        return pd.DataFrame()
