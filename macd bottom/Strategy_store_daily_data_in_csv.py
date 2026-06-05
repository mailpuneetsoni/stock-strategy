import pandas as pd
import yfinance as yf
from tqdm import tqdm
import os

# Step 1: Read tickers from equity_500.csv on the Desktop
equity_file = r'D:\Stock_Strategy\filtered_nifty_market_cap_stocks.csv'

if not os.path.exists(equity_file):
    raise FileNotFoundError(f"Could not find csv at {equity_file}")

try:
    equity_df = pd.read_csv(equity_file)
except Exception as e:
    raise ValueError(f"Failed to read filtered_nifty_market_cap_stocks.csv: {e}")

tickers = [symbol for symbol in equity_df['Ticker'].tolist()]

# Step 2: Download DAILY data for the last ~2 years
try:
    data = yf.download(
        tickers,
        start='2025-06-03',
        end='2026-06-05',
        interval='1d',          # <-- daily bars
        group_by='ticker',
        threads=True,
        progress=True
    )

except Exception as e:
    print(f"Bulk download failed: {e}. Falling back to individual downloads.")
    data_dict = {}
    for ticker in tqdm(tickers):
        try:
            data_dict[ticker] = yf.download(
                ticker,
                start='2025-06-03',
                end='2026-06-05',
                interval='1d',
                progress=False
            )
        except Exception as ticker_error:
            print(f"Error retrieving data for {ticker}: {ticker_error}. Skipping.")
            continue
    if data_dict:
        data = pd.concat(data_dict, axis=1)
    else:
        raise ValueError("No data retrieved for any ticker.")

# Step 3: Clean the data
if isinstance(data, pd.DataFrame):
    data = data.reset_index()
    data = data.dropna(axis=1, how='all')
    data = data.dropna(axis=0, how='all')
    data = data.ffill()
else:
    raise ValueError("No valid data retrieved to process.")

# Step 4: Save to stock_data_daily.csv on the Desktop
output_file = os.path.join('D:\Stock_Strategy\stock_data_daily.csv') 
try:
    data.to_csv(output_file, index=False)
    print(f"Daily data successfully saved to {output_file}")
except Exception as e:
    print(f"Error writing to {output_file}: {e}")
    alt_output_file = os.path.join('D:\Stock_Strategy\stock_data_daily_backup.csv')
    try:
        data.to_csv(alt_output_file, index=False)
        print(f"Data saved to alternative file: {alt_output_file}")
    except Exception as alt_e:
        raise ValueError(f"Unable to save data: {alt_e}")