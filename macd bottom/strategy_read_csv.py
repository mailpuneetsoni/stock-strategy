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
                    ohlcv_data.columns = pd.MultiIndex.from_product([[ticker], required_columns +                                                             ['MACD', 'Smoothed_MACD', 'Signal', 'Histogram',                                                           'RSI', 'SMA_50', 'SMA_200', 'Volume_Avg']])
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

# Step 5: Find local minima with RSI and Volume filters (MODIFIED)
def find_local_minima(series, data_df, ticker, base_order=5, std_factor=0.5, min_gap=5, 
                      require_macd_negative=False, rsi_threshold=30, volume_multiplier=1.5, rolling_window=20):

    if series.empty or len(series) < base_order * 2:
        return []

    order = max(base_order, len(series) // 50)
    minima_indices = argrelextrema(series.values, np.less, order=order)[0]
    minima_dates = series.index[minima_indices]

    if minima_dates.empty:
        return []

    # MODIFIED: Use rolling mean and std instead of expanding
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

# Step 6: Find local minima and evaluate TP, FP, TN, FN
minima_results = []
evaluation_results = []
look_ahead_days = 5  # Approx. 1 trading week; set to 10 for 2 weeks
price_threshold = 0.02
for ticker in cleaned_data.columns.get_level_values(0).unique():
    try:
        smoothed_macd_series = cleaned_data[(ticker, 'Smoothed_MACD')].dropna()
        close_series = cleaned_data[(ticker, 'Close')].dropna()
        if len(smoothed_macd_series) > 2:
            # MODIFIED: Added rolling_window parameter
            minima_dates = find_local_minima(smoothed_macd_series, cleaned_data, ticker, 
                                           base_order=5, std_factor=0.5, min_gap=5, 
                                           require_macd_negative=False, rsi_threshold=30, volume_multiplier=1.2,
                                           rolling_window=20)
            if not minima_dates:
                print(f"{ticker}: No minima found.")
            all_dates = smoothed_macd_series.index
            minima_set = set(minima_dates)
            for date in all_dates:
                try:
                    current_index = close_series.index.get_loc(date)
                    if current_index + look_ahead_days >= len(close_series):
                        continue
                    current_close = close_series.iloc[current_index]
                    future_closes = close_series.iloc[current_index + 1:current_index + look_ahead_days + 1]
                    max_future_close = future_closes.max()
                    price_increase = (max_future_close - current_close) / current_close
                    is_positive = price_increase >= price_threshold
                    is_predicted = date in minima_set
                    sma_50_val = cleaned_data[(ticker, 'SMA_50')].loc[date]
                    sma_200_val = cleaned_data[(ticker, 'SMA_200')].loc[date]
                    trend_valid = current_close > sma_200_val and sma_50_val > sma_200_val
                    if is_predicted and is_positive and trend_valid:
                        outcome = 'TP'
                    elif is_predicted and (not is_positive or not trend_valid):
                        outcome = 'FP'
                    elif not is_predicted and not is_positive:
                        outcome = 'TN'
                    elif not is_predicted and is_positive:
                        outcome = 'FN'
                    if is_predicted and trend_valid:
                        minima_results.append({
                            'Date': date,
                            'Ticker': ticker,
                            'Smoothed_MACD': smoothed_macd_series.loc[date],
                            'RSI': cleaned_data[(ticker, 'RSI')].loc[date],
                            'Volume': cleaned_data[(ticker, 'Volume')].loc[date],
                            'Volume_Avg': cleaned_data[(ticker, 'Volume_Avg')].loc[date],
                            'SMA_50': sma_50_val,
                            'SMA_200': sma_200_val
                        })
                    evaluation_results.append({
                        'Date': date,
                        'Ticker': ticker,
                        'Smoothed_MACD': smoothed_macd_series.loc[date] if is_predicted and trend_valid else np.nan,
                        'Close': current_close,
                        'Max_Future_Close': max_future_close,
                        'Price_Increase': price_increase,
                        'RSI': cleaned_data[(ticker, 'RSI')].loc[date],
                        'Volume': cleaned_data[(ticker, 'Volume')].loc[date],
                        'Volume_Avg': cleaned_data[(ticker, 'Volume_Avg')].loc[date],
                        'SMA_50': sma_50_val,
                        'SMA_200': sma_200_val,
                        'Outcome': outcome
                    })
                except Exception as e:
                    print(f"Error evaluating {ticker} at {date}: {e}")
                    continue
        else:
            print(f"{ticker}: Insufficient data for minima detection.")
    except Exception as e:
        print(f"Error finding minima for {ticker}: {e}")
        continue

if not cleaned_data.empty:
    print("Minima detection and evaluation completed.")
else:
    print("No valid OHLCV or MACD data found for any ticker.")
    cleaned_data = pd.DataFrame()

# Step 7: Save minima results
output_file = os.path.join(desktop_path, 'macd_minima.csv')
try:
    if minima_results:
        minima_df = pd.DataFrame(minima_results)
        minima_df = minima_df.sort_values(by='Date').reset_index(drop=True)
        minima_df['Smoothed_MACD'] = minima_df['Smoothed_MACD'].round(4)
        minima_df['RSI'] = minima_df['RSI'].round(2)
        minima_df['Volume'] = minima_df['Volume'].round(2)
        minima_df['Volume_Avg'] = minima_df['Volume_Avg'].round(2)
        minima_df['SMA_50'] = minima_df['SMA_50'].round(2)
        minima_df['SMA_200'] = minima_df['SMA_200'].round(2)
        minima_df.to_csv(output_file, index=False)
        print(f"All local minima saved to {output_file}")
    else:
        pd.DataFrame(columns=['Date', 'Ticker', 'Smoothed_MACD', 'RSI', 'Volume', 'Volume_Avg', 'SMA_50', 'SMA_200']).to_csv(output_file, index=False)
        print(f"No minima found. Empty CSV saved to {output_file}")
except Exception as e:
    print(f"Error saving minima results: {e}")

# Step 8: Save evaluation results and print metrics
eval_output_file = os.path.join(desktop_path, 'evaluation_metrics.csv')
try:
    if evaluation_results:
        eval_df = pd.DataFrame(evaluation_results)
        eval_df = eval_df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)
        eval_df['Smoothed_MACD'] = eval_df['Smoothed_MACD'].round(4)
        eval_df['Close'] = eval_df['Close'].round(2)
        eval_df['Max_Future_Close'] = eval_df['Max_Future_Close'].round(2)
        eval_df['Price_Increase'] = eval_df['Price_Increase'].round(4)
        eval_df['RSI'] = eval_df['RSI'].round(2)
        eval_df['Volume'] = eval_df['Volume'].round(2)
        eval_df['Volume_Avg'] = eval_df['Volume_Avg'].round(2)
        eval_df['SMA_50'] = eval_df['SMA_50'].round(2)
        eval_df['SMA_200'] = eval_df['SMA_200'].round(2)
        eval_df.to_csv(eval_output_file, index=False)
        outcome_counts = eval_df['Outcome'].value_counts()
        tp = outcome_counts.get('TP', 0)
        fp = outcome_counts.get('FP', 0)
        tn = outcome_counts.get('TN', 0)
        fn = outcome_counts.get('FN', 0)
        print("Evaluation Metrics:")
        print(f"True Positives (TP): {tp}")
        print(f"False Positives (FP): {fp}")
        print(f"True Negatives (TN): {tn}")
        print(f"False Negatives (FN): {fn}")
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
            print(f"Precision: {precision:.4f}")
        if (tp + fn) > 0:
            recall = tp / (tp + fn)
            print(f"Recall: {recall:.4f}")
        print(f"Results saved to {eval_output_file}")
    else:
        pd.DataFrame(columns=['Date', 'Ticker', 'Smoothed_MACD', 'Close', 'Max_Future_Close', 'Price_Increase', 
                              'RSI', 'Volume', 'Volume_Avg', 'SMA_50', 'SMA_200', 'Outcome']).to_csv(eval_output_file, index=False)
        print(f"No evaluation results. Empty CSV saved to {eval_output_file}")
except Exception as e:
    print(f"Error saving evaluation results: {e}")

# Step 9: Calculate and print total execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Total execution time: {execution_time:.2f} seconds")