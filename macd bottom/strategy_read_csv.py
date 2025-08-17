import pandas as pd
import numpy as np
import os
import talib
import time
from scipy.stats import linregress

start_time = time.time()

# Step 1: Define the path to stock_data.csv on the Desktop
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
file_path = os.path.join(desktop_path, 'stock_data.csv')

# Step 2: Read the CSV file with error handling
try:
    data = pd.read_csv(file_path, header=[0, 1], index_col=0, parse_dates=True)
    print(f"Data successfully loaded from {file_path}")
except FileNotFoundError:
    print(f"Error: {file_path} not found on Desktop.")
    raise
except Exception as e:
    print(f"Error loading stock_data.csv: {e}")
    raise ValueError(f"Failed to read {file_path}. Ensure the file is a valid CSV with the expected format.")

# Step 3: Extract unique tickers from the multi-level columns
tickers = data.columns.get_level_values(0).unique()
if not tickers.size:
    raise ValueError("No tickers found in the data.")

# Step 4: Clean the data and compute MACD
required_columns = ['Close']  # Only need Close for MACD
cleaned_data = pd.DataFrame()
for ticker in tickers:
    try:
        ticker_columns = [(ticker, col) for col in required_columns]
        if all(col in data.columns for col in ticker_columns):
            ohlcv_data = data[ticker][required_columns].copy()
            ohlcv_data = ohlcv_data.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()
            if not ohlcv_data.empty:
                close = ohlcv_data['Close'].values
                if len(close) >= 26:  # Minimum for MACD calculation
                    # Calculate MACD
                    macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
                    macd_series = pd.Series(macd, index=ohlcv_data.index[-len(macd):])
                    min_len = min(len(macd), len(ohlcv_data))
                    # Create DataFrame
                    ticker_data = pd.DataFrame({
                        'Close': ohlcv_data['Close'].values[-min_len:],
                        'MACD': macd_series.values[-min_len:]
                    }, index=ohlcv_data.index[-min_len:])
                    ticker_data['Ticker'] = ticker
                    cleaned_data = pd.concat([cleaned_data, ticker_data], axis=0)
                else:
                    print(f"Insufficient data for {ticker} to compute MACD (need at least 26 periods).")
            else:
                print(f"No valid Close data for {ticker} after cleaning.")
        else:
            missing_columns = [col for col in ticker_columns if not col in data.columns]
            print(f"Skipping {ticker}: Missing columns {missing_columns}")
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        continue

# Step 5: Sort the data by date in ascending order
if not cleaned_data.empty:
    cleaned_data = cleaned_data.sort_index(ascending=True)
else:
    print("Warning: No data found after processing.")

# Step 6: Save the sorted data to a CSV file
output_file_path = os.path.join(desktop_path, 'cleaned_stock_data.csv')
try:
    if not cleaned_data.empty:
        cleaned_data.reset_index().rename(columns={'index': 'Date'})[['Date', 'Ticker', 'Close', 'MACD']].to_csv(output_file_path, index=False)
    else:
        pd.DataFrame(columns=['Date', 'Ticker', 'Close', 'MACD']).to_csv(output_file_path, index=False)
    print(f"Sorted data saved to {output_file_path}")
except Exception as e:
    print(f"Error saving data to {output_file_path}: {e}")

# Step 7: Calculate and print total execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Total execution time: {execution_time:.2f} seconds")