import pandas as pd
import numpy as np
import os
import talib
from scipy.signal import argrelextrema
import time
import hmmlearn.hmm

# Record start time
start_time = time.time()

# Step 1: Define the path to stock_data.csv on the Desktop
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
file_path = os.path.join(desktop_path, 'stock_data.csv')

# Step 2: Read the CSV file with error handling
try:
    # Read CSV, assuming multi-level columns (ticker, attribute) and Date as index
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

# Step 4: Clean the data and compute MACD using ta-lib
required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
cleaned_data = pd.DataFrame()
for ticker in tickers:
    try:
        # Check if all OHLCV columns exist for the ticker
        ticker_columns = [(ticker, col) for col in required_columns]
        if all(col in data.columns for col in ticker_columns):
            # Extract OHLCV data
            ohlcv_data = data[ticker][required_columns].copy()
            # Remove non-numeric or NaN values
            ohlcv_data = ohlcv_data.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()
            if not ohlcv_data.empty:
                # Compute MACD using ta-lib (standard parameters: fast=12, slow=26, signal=9)
                close = ohlcv_data['Close'].values
                if len(close) >= 26:  # Ensure enough data for MACD calculation
                    macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
                    assert len(macd) == len(ohlcv_data), f"MACD length mismatch for {ticker}: {len(macd)} != {len(ohlcv_data)}"
                    # Smooth MACD with a rolling mean (window=5 to reduce noise)
                    macd_series = pd.Series(macd, index=ohlcv_data.index[-len(macd):])
                    smoothed_macd = macd_series.rolling(window=5, min_periods=1).mean().values
                    # Create DataFrame with MACD data
                    macd_data = pd.DataFrame({
                        'MACD': macd,
                        'Smoothed_MACD': smoothed_macd,
                        'Signal': signal,
                        'Histogram': hist
                    }, index=ohlcv_data.index[-len(macd):])
                    # Combine OHLCV and MACD data
                    original_len = len(ohlcv_data)
                    ohlcv_data = ohlcv_data.join(macd_data, how='inner')  # Keep only rows with valid MACD
                    assert len(ohlcv_data) == original_len, f"Data loss in join for {ticker}: {len(ohlcv_data)} != {original_len}"
                    # Add to cleaned_data with multi-level columns
                    ohlcv_data.columns = pd.MultiIndex.from_product([[ticker], required_columns + ['MACD', 'Smoothed_MACD', 'Signal', 'Histogram']])
                    cleaned_data = pd.concat([cleaned_data, ohlcv_data], axis=1)
                else:
                    print(f"Insufficient data for {ticker} to compute MACD (need at least 26 periods).")
            else:
                print(f"No valid OHLCV data for {ticker} after cleaning.")
        else:
            missing_columns = [col for col in ticker_columns if not col in data.columns]
            print(f"Skipping {ticker}: Missing columns {missing_columns}")
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        continue

def find_local_minima(series, base_order=5, std_factor=0.5, min_gap=5, require_macd_negative=False):
    """
    Find robust local minima in a series.
    
    Parameters:
    - series (pd.Series): Time series data (e.g., Smoothed_MACD).
    - base_order (int): Base lookback for minima detection.
    - std_factor (float): How far below rolling mean a minima must be (in std dev units).
    - min_gap (int): Minimum number of bars between two minima.
    - require_macd_negative (bool): If True, keep only minima where value < 0 (disabled for broader contrarian detection).
    
    Returns:
    - List of minima dates.
    """
    if series.empty or len(series) < base_order * 2:
        return []

    # Dynamic order scaling with data length
    order = max(base_order, len(series) // 50)

    # Step 1: Find raw minima
    minima_indices = argrelextrema(series.values, np.less, order=order)[0]
    minima_dates = series.index[minima_indices]

    if minima_dates.empty:
        return []

    # Step 2: Magnitude filtering (only significant dips)
    rolling_mean = series.rolling(20, min_periods=5).mean()
    rolling_std = series.rolling(20, min_periods=5).std()
    valid_minima = [
        d for d in minima_dates
        if series.loc[d] < (rolling_mean.loc[d] - std_factor * rolling_std.loc[d])
    ]

    # Step 3: Optional: Require MACD < 0 (bullish reversal filter; disabled to capture all extremes)
    if require_macd_negative:
        valid_minima = [d for d in valid_minima if series.loc[d] < 0]

    # Step 4: Merge nearby minima, keep the deepest
    filtered_minima = []
    for date in valid_minima:
        if not filtered_minima or (date - filtered_minima[-1]).days > min_gap:
            filtered_minima.append(date)
        else:
            # Replace with deeper minima
            if series.loc[date] < series.loc[filtered_minima[-1]]:
                filtered_minima[-1] = date

    return filtered_minima

# Step 5: Find local minima and apply HMM for state probabilities
minima_results = []
if not cleaned_data.empty:
    for ticker in cleaned_data.columns.get_level_values(0).unique():
        try:
            # Use smoothed MACD for minima detection and HMM
            smoothed_macd_series = cleaned_data[(ticker, 'Smoothed_MACD')].dropna()
            if len(smoothed_macd_series) > 2:  # Need at least 3 points for minima
                # Find all minima on smoothed MACD (without requiring negative values)
                minima_dates = find_local_minima(smoothed_macd_series, base_order=5, std_factor=0.5, min_gap=5, require_macd_negative=False)
                if not minima_dates:
                    print(f"{ticker}: No minima found.")
                    continue
                # Only train HMM if minima are found and sufficient data exists
                if len(smoothed_macd_series) >= 50:  # Minimum data points for HMM
                    try:
                        # Prepare data for HMM (2D array for hmmlearn)
                        X = smoothed_macd_series.values.reshape(-1, 1)
                        # Initialize and fit Gaussian HMM with 2 states
                        hmm_model = hmmlearn.hmm.GaussianHMM(
                            n_components=2,
                            covariance_type='diag',
                            n_iter=100,
                            tol=1e-4,
                            random_state=42  # For reproducibility
                        )
                        hmm_model.fit(X)
                        # Predict posterior probabilities for all data points
                        state_probs = hmm_model.predict_proba(X)
                        # Get state means and assign labels
                        state_means = hmm_model.means_.flatten()
                        state_labels = [''] * 2
                        sorted_indices = np.argsort(state_means)
                        state_labels[sorted_indices[0]] = 'bearish'  # Lowest mean
                        state_labels[sorted_indices[1]] = 'bullish'  # Highest mean
                        # Find bearish state index for buy signals (reversal logic: buy at bearish extremes)
                        bearish_idx = [i for i, label in enumerate(state_labels) if label == 'bearish'][0]
                        # Map minima dates to indices in smoothed_macd_series
                        minima_indices = [smoothed_macd_series.index.get_loc(date) for date in minima_dates if date in smoothed_macd_series.index]
                        # Append results with bearish state probability (higher = stronger buy signal)
                        for idx, date in zip(minima_indices, minima_dates):
                            minima_results.append({
                                'Date': date,
                                'Ticker': ticker,
                                'MACD': smoothed_macd_series.loc[date],  # Use smoothed MACD
                                'Bearish_State_Probability': state_probs[idx, bearish_idx]  # Probability of bearish state
                            })
                    except Exception as e:
                        print(f"Error training HMM for {ticker}: {e}")
                        # Append minima with NaN probability
                        for date in minima_dates:
                            minima_results.append({
                                'Date': date,
                                'Ticker': ticker,
                                'MACD': smoothed_macd_series.loc[date],
                                'Bearish_State_Probability': np.nan
                            })
                else:
                    print(f"Insufficient data for HMM training for {ticker} (<50 points).")
                    # Append minima with NaN probability
                    for date in minima_dates:
                        minima_results.append({
                            'Date': date,
                            'Ticker': ticker,
                            'MACD': smoothed_macd_series.loc[date],
                            'Bearish_State_Probability': np.nan
                        })
            else:
                print(f"{ticker}: Insufficient data for minima detection.")
        except Exception as e:
            print(f"Error finding minima for {ticker}: {e}")
            continue
else:
    print("No valid OHLCV or MACD data found for any ticker.")
    cleaned_data = pd.DataFrame()  # Ensure cleaned_data is an empty DataFrame if no valid data

# Step 6: Save all local minima to a readable CSV file on the Desktop
output_file = os.path.join(desktop_path, 'macd_minima.csv')
try:
    # Create DataFrame from minima results
    if minima_results:
        minima_df = pd.DataFrame(minima_results)
        # Sort by Date for readability
        minima_df = minima_df.sort_values(by='Date').reset_index(drop=True)
        # Format MACD and Bearish_State_Probability to 4 decimal places for clarity
        minima_df['MACD'] = minima_df['MACD'].round(4)
        minima_df['Bearish_State_Probability'] = minima_df['Bearish_State_Probability'].round(4)
        minima_df.to_csv(output_file, index=False)
        print(f"All local minima saved to {output_file}")
    else:
        # Save an empty CSV with headers if no minima are found
        pd.DataFrame(columns=['Date', 'Ticker', 'MACD', 'Bearish_State_Probability']).to_csv(output_file, index=False)
        print(f"No minima found. Empty CSV saved to {output_file}")
except Exception as e:
    print(f"Error saving to {output_file}: {e}")
    # Try alternative file
    alt_output_file = os.path.join(desktop_path, 'macd_minima_backup.csv')
    try:
        minima_df.to_csv(alt_output_file, index=False) if minima_results else pd.DataFrame(columns=['Date', 'Ticker', 'MACD', 'Bearish_State_Probability']).to_csv(alt_output_file, index=False)
        print(f"Local minima saved to alternative file: {alt_output_file}")
    except Exception as alt_e:
        print(f"Failed to save to {alt_output_file}: {alt_e}")
        raise ValueError("Unable to save MACD minima to CSV file.")

# Calculate and print total execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Total execution time: {execution_time:.2f} seconds")

# The cleaned_data DataFrame is now available for further use
# Example: print(cleaned_data.shape) to check dimensions