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

# Step 4: Define 2D Kalman Filter class for smoothing
class KalmanFilter2D:
    """
    A simple 2D Kalman Filter for smoothing a 1D time series using a position-velocity model.
    State vector: [position, velocity]
    """
    def __init__(self, dt=1.0, Q_scale=0.001, R_scale=0.1, initial_position=0.0, initial_velocity=0.0, initial_covariance=1.0):
        """
        Initialize the Kalman Filter.
        Parameters:
        - dt: Time step (default: 1.0 for weekly data)
        - Q_scale: Process noise covariance scale (controls model uncertainty)
        - R_scale: Measurement noise covariance (controls trust in measurements)
        - initial_position: Initial position estimate (default: 0.0)
        - initial_velocity: Initial velocity estimate (default: 0.0)
        - initial_covariance: Initial state covariance (default: 1.0)
        """
        # State vector: [position, velocity]
        self.x = np.array([[initial_position], [initial_velocity]])
        # State transition matrix: F = [[1, dt], [0, 1]]
        self.F = np.array([[1.0, dt], [0.0, 1.0]])
        # Observation matrix: H = [1, 0] (observe position only)
        self.H = np.array([[1.0, 0.0]])
        # Process noise covariance: Q
        self.Q = Q_scale * np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])
        # Measurement noise covariance: R
        self.R = np.array([[R_scale]])
        # State covariance matrix: P
        self.P = initial_covariance * np.eye(2)
        # Identity matrix
        self.I = np.eye(2)

    def predict(self):
        """Predict the next state and covariance."""
        # Predict state: x = F * x
        self.x = self.F @ self.x
        # Predict covariance: P = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """Update the state with a new measurement."""
        if np.isnan(z):
            return  # Skip NaN measurements
        z = np.array([[z]])
        # Innovation: y = z - H * x
        y = z - self.H @ self.x
        # Innovation covariance: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + self.R
        # Kalman gain: K = P * H^T * S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # Update state: x = x + K * y
        self.x = self.x + K @ y
        # Update covariance: P = (I - K * H) * P
        self.P = (self.I - K @ self.H) @ self.P

    def smooth(self, series):
        """
        Smooth a time series using the Kalman Filter.
        Parameters:
        - series: 1D array-like (e.g., MACD values)
        Returns:
        - Smoothed position values
        """
        series = np.array(series)
        smoothed = []
        for z in series:
            self.predict()
            self.update(z)
            smoothed.append(self.x[0, 0])  # Extract position
        return np.array(smoothed)

# Step 5: Clean the data, compute MACD using ta-lib, and smooth the MACD line
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
                    # Smooth the MACD line using 2D Kalman Filter
                    kf = KalmanFilter2D(dt=1.0, Q_scale=0.001, R_scale=0.1)
                    smoothed_macd = kf.smooth(macd)
                    # Align MACD and smoothed data with the DataFrame index
                    macd_data = pd.DataFrame({
                        'MACD': macd,
                        'Smoothed_MACD': smoothed_macd,
                        'Signal': signal,
                        'Histogram': hist
                    }, index=ohlcv_data.index[-len(macd):])
                    # Combine OHLCV and MACD data
                    ohlcv_data = ohlcv_data.join(macd_data, how='inner')  # Keep only rows with valid MACD
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

# Step 6: Find local minima of the smoothed MACD line for each ticker and compute HMM probabilities
def find_local_minima(series, order=1):
    """
    Find local minima in a series using scipy's argrelextrema.
    - series: A pandas Series with a DatetimeIndex.
    - order: How many points on each side to use for comparison.
    Returns the indices (dates) of the local minima.
    """
    minima_indices = argrelextrema(series.values, np.less, order=order)[0]
    return series.index[minima_indices]

# Initialize a list to store minima for all tickers
minima_results = []
if not cleaned_data.empty:
    for ticker in cleaned_data.columns.get_level_values(0).unique():
        try:
            # Use smoothed MACD for minima detection and HMM
            smoothed_macd_series = cleaned_data[(ticker, 'Smoothed_MACD')].dropna()
            if len(smoothed_macd_series) > 2:  # Need at least 3 points for minima
                # Find all minima on smoothed MACD
                minima_dates = find_local_minima(smoothed_macd_series, order=1)
                if not minima_dates.empty:
                    # Only train HMM if minima are found and sufficient data exists
                    if len(smoothed_macd_series) >= 50:  # Minimum data points for HMM
                        try:
                            # Prepare data for HMM (2D array for hmmlearn)
                            X = smoothed_macd_series.values.reshape(-1, 1)
                            # Initialize and fit Gaussian HMM with 4 states
                            hmm_model = hmmlearn.hmm.GaussianHMM(
                                n_components=4,
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
                            state_labels = [''] * 4
                            sorted_indices = np.argsort(state_means)
                            state_labels[sorted_indices[0]] = 'bearish'  # Lowest mean
                            state_labels[sorted_indices[3]] = 'bullish'  # Highest mean
                            state_labels[sorted_indices[2]] = 'trend reversal'  # Second-highest
                            # Find state closest to zero mean
                            zero_diff = np.abs(state_means)
                            state_labels[np.argmin(zero_diff)] = 'consolidation'
                            # Find bearish state index
                            bearish_idx = [i for i, label in enumerate(state_labels) if label == 'bearish'][0]
                            # Map minima dates to indices in smoothed_macd_series
                            minima_indices = [smoothed_macd_series.index.get_loc(date) for date in minima_dates if date in smoothed_macd_series.index]
                            # Append results with bearish state probability
                            for idx, date in zip(minima_indices, minima_dates):
                                minima_results.append({
                                    'Date': date,
                                    'Ticker': ticker,
                                    'MACD': smoothed_macd_series.loc[date],  # Use smoothed MACD
                                    'State_Probability': state_probs[idx, bearish_idx]
                                })
                        except Exception as e:
                            print(f"Error training HMM for {ticker}: {e}")
                            # Append minima with NaN probability
                            for date in minima_dates:
                                minima_results.append({
                                    'Date': date,
                                    'Ticker': ticker,
                                    'MACD': smoothed_macd_series.loc[date],
                                    'State_Probability': np.nan
                                })
                    else:
                        print(f"Insufficient data for HMM training for {ticker} (<50 points).")
                        # Append minima with NaN probability
                        for date in minima_dates:
                            minima_results.append({
                                'Date': date,
                                'Ticker': ticker,
                                'MACD': smoothed_macd_series.loc[date],
                                'State_Probability': np.nan
                            })
            else:
                print(f"{ticker}: Insufficient data for minima detection.")
        except Exception as e:
            print(f"Error finding minima for {ticker}: {e}")
            continue
else:
    print("No valid OHLCV or MACD data found for any ticker.")
    cleaned_data = pd.DataFrame()  # Ensure cleaned_data is an empty DataFrame if no valid data

# Step 7: Save all local minima to a readable CSV file on the Desktop
output_file = os.path.join(desktop_path, 'macd_minima.csv')
try:
    # Create DataFrame from minima results
    if minima_results:
        minima_df = pd.DataFrame(minima_results)
        # Sort by Date for readability
        minima_df = minima_df.sort_values(by='Date').reset_index(drop=True)
        # Format MACD and State_Probability to 4 decimal places for clarity
        minima_df['MACD'] = minima_df['MACD'].round(4)
        minima_df['State_Probability'] = minima_df['State_Probability'].round(4)
        minima_df.to_csv(output_file, index=False)
        print(f"All local minima saved to {output_file}")
    else:
        # Save an empty CSV with headers if no minima are found
        pd.DataFrame(columns=['Date', 'Ticker', 'MACD', 'State_Probability']).to_csv(output_file, index=False)
        print(f"No minima found. Empty CSV saved to {output_file}")
except Exception as e:
    print(f"Error saving to {output_file}: {e}")
    # Try alternative file
    alt_output_file = os.path.join(desktop_path, 'macd_minima_backup.csv')
    try:
        minima_df.to_csv(alt_output_file, index=False) if minima_results else pd.DataFrame(columns=['Date', 'Ticker', 'MACD', 'State_Probability']).to_csv(alt_output_file, index=False)
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