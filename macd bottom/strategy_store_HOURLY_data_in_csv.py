import pandas as pd
import yfinance as yf
from tqdm import tqdm
import os
import datetime

# Step 1: Read the tickers from EQUITY_L.csv located on the Desktop
equity_file = r'D:\Stock_Strategy\filtered_nifty_market_cap_stocks.csv'
# Check if the file exists
if not os.path.exists(equity_file):
    raise FileNotFoundError(f"Could not find csv at {equity_file}")

# Read the CSV, assuming it has a column named 'Ticker' for ticker symbols
try:
    equity_df = pd.read_csv(equity_file)
except Exception as e:
    raise ValueError(f"Failed to read filtered_nifty_market_cap_stocks.csv: {e}")

# Extract tickers (add '.NS' suffix for NSE tickers as required by yfinance)
tickers = [symbol for symbol in equity_df['Ticker'].tolist()]

# Step 2: Retrieve weekly historical data for the last 25 years using yfinance bulk download
# Dates: From 2000-08-06 to 2025-08-06, with interval='1wk' for weekly data

try:
    data = yf.download(tickers, start='2025-09-03', end='2026-06-05', interval='1h', group_by='ticker', threads=True, progress=True)
    
    if data is None or getattr(data, "empty", True):
        raise ValueError("Downloaded data is empty or null")

    print("data type:", type(data))
    print("data format:", getattr(data, "shape", None))
    print("data columns:", data.columns if hasattr(data, "columns") else None)
    print("data head:")
    print(data.head())
    
    # If bulk download fails, fall back to individual downloads
except Exception as e:
    print(f"Bulk download failed: {e}. Falling back to individual downloads.")
    data_dict = {}
    for ticker in tqdm(tickers):
        try:
            data_dict[ticker] = yf.download(ticker, start='2025-09-03', end='2026-06-05', interval='1h', progress=False)
        except Exception as ticker_error:
            print(f"Error retrieving data for {ticker}: {ticker_error}. Skipping.")
            continue
    if data_dict:
        data = pd.concat(data_dict, axis=1)
    else:
        raise ValueError("No data retrieved for any ticker.")

# Step 3: Necessary data handling
# - Reset index to make 'Date' a column
# - Drop columns/rows that are entirely NaN
# - Forward-fill missing values
if isinstance(data, pd.DataFrame):
    data = data.reset_index()  # Make Date a column
    data = data.dropna(axis=1, how='all')  # Drop columns (tickers) with all NaN
    data = data.dropna(axis=0, how='all')  # Drop rows with all NaN
    data = data.ffill()  # Forward-fill missing values
else:
    raise ValueError("No valid data retrieved to process.")

# Step 4: Store the data in a CSV file explicitly on the Desktop
output_file = os.path.join('D:\Stock_Strategy\input_stock_data_hourly.csv')
try:
    data.to_csv(output_file, index=False)
    print(f"Data successfully saved to {output_file}")
except Exception as e:
    print(f"Error writing to {output_file}: {e}")
    # Attempt to save to an alternative file to avoid data loss
    alt_output_file = os.path.join('D:\Stock_Strategy\stock_data_backup.csv')
    try:
        data.to_csv(alt_output_file, index=False)
        print(f"Data saved to alternative file: {alt_output_file}")
    except Exception as alt_e:
        print(f"Failed to save to alternative file {alt_output_file}: {alt_e}")
        raise ValueError("Unable to save data to CSV file.")