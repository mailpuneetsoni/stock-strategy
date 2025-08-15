import pandas as pd
import numpy as np
import os
import talib
from scipy.signal import argrelextrema
import time

start_time = time.time()

# Step 1: Define the path to stock_data.csv on the Desktop
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
file_path = os.path.join(desktop_path, 'stock_data.csv')

# Step 2: Read the CSV file with error handling and compute time range
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

# Step 4: Clean the data and compute MACD, RSI, SMAs, and Volume Average
required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
cleaned_data = pd.DataFrame()
for ticker in tickers:
    try:
        ticker_columns = [(ticker, col) for col in required_columns]
        if all(col in data.columns for col in ticker_columns):
            ohlcv_data = data[ticker][required_columns].copy()
            ohlcv_data = ohlcv_data.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()
            if not ohlcv_data.empty:
                close = ohlcv_data['Close'].values
                volume = ohlcv_data['Volume'].values
                if len(close) >= 200:
                    macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
                    macd_series = pd.Series(macd, index=ohlcv_data.index[-len(macd):])
                    smoothed_macd = macd_series.rolling(window=5, min_periods=1).mean().values
                    rsi = talib.RSI(close, timeperiod=14)
                    sma_50 = talib.SMA(close, timeperiod=50)
                    sma_200 = talib.SMA(close, timeperiod=200)
                    volume_avg = talib.SMA(volume, timeperiod=20)
                    min_len = min(len(macd), len(rsi), len(sma_50), len(sma_200), len(volume_avg), len(ohlcv_data))
                    macd_data = pd.DataFrame({
                        'MACD': macd[-min_len:],
                        'Smoothed_MACD': smoothed_macd[-min_len:],
                        'Signal': signal[-min_len:],
                        'Histogram': hist[-min_len:],
                        'RSI': rsi[-min_len:],
                        'SMA_50': sma_50[-min_len:],
                        'SMA_200': sma_200[-min_len:],
                        'Volume_Avg': volume_avg[-min_len:]
                    }, index=ohlcv_data.index[-min_len:])
                    ohlcv_data = ohlcv_data.join(macd_data, how='inner')
                    ohlcv_data.columns = pd.MultiIndex.from_product([[ticker], required_columns + 
                                                                    ['MACD', 'Smoothed_MACD', 'Signal', 'Histogram', 
                                                                     'RSI', 'SMA_50', 'SMA_200', 'Volume_Avg']])
                    cleaned_data = pd.concat([cleaned_data, ohlcv_data], axis=1)
                else:
                    print(f"Insufficient data for {ticker} to compute indicators (need at least 200 periods).")
            else:
                print(f"No valid OHLCV data for {ticker} after cleaning.")
        else:
            missing_columns = [col for col in ticker_columns if not col in data.columns]
            print(f"Skipping {ticker}: Missing columns {missing_columns}")
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        continue

# Step 5: Find local minima with RSI and Volume filters
def find_local_minima(series, data_df, ticker, base_order=5, std_factor=0.5, min_gap=3, 
                      require_macd_negative=False, rsi_threshold=30, volume_multiplier=1.5, rolling_window=20):
    if series.empty or len(series) < base_order * 2:
        return []

    order = max(base_order, len(series) // 50)
    minima_indices = argrelextrema(series.values, np.less, order=order)[0]
    minima_dates = series.index[minima_indices]

    if minima_dates.empty:
        return []

    rolling_mean = series.rolling(window=rolling_window, min_periods=5).mean()
    rolling_std = series.rolling(window=rolling_window, min_periods=5).std()
    valid_minima = [
        d for d in minima_dates
        if series.loc[d] < (rolling_mean.loc[d] - std_factor * rolling_std.loc[d])
    ]

    if require_macd_negative:
        valid_minima = [d for d in valid_minima if series.loc[d] < 0]

    filtered_minima = []
    for d in valid_minima:
        rsi_val = data_df[(ticker, 'RSI')].loc[d]
        volume_val = data_df[(ticker, 'Volume')].loc[d]
        volume_avg_val = data_df[(ticker, 'Volume_Avg')].loc[d]
        if rsi_val < rsi_threshold and volume_val > (volume_multiplier * volume_avg_val):
            if not filtered_minima or (d - filtered_minima[-1]).days > min_gap:
                filtered_minima.append(d)
            else:
                if series.loc[d] < series.loc[filtered_minima[-1]]:
                    filtered_minima[-1] = d

    return filtered_minima

# Step 9: Calculate and print total execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Total execution time: {execution_time:.2f} seconds")