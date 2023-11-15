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

# Example usage
if __name__ == "__main__":
    stock_symbol = 'AAPL'  
    start_date = '2020-01-01'
    end_date = '2023-01-01'

    data = fetch_data(stock_symbol, start_date, end_date)
    print(data.head())  # Print the first few rows of the fetched data
