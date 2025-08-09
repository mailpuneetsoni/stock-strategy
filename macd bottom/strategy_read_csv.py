import pandas as pd
import numpy as np
import os
import talib
from datetime import datetime, timedelta

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

# Step 6: Find local minima of the MACD line for each ticker within the past 1 month
def find_local_minima(series):
    """
    Find local minima in a series where value < previous and value < next.
    Returns the indices (dates) of minima.
    """
    minima = (series.shift(1) > series) & (series.shift(-1) > series)
    return series.index[minima]

# Define the date range for the past 1 month
end_date = pd.to_datetime('2025-08-06')
start_date = end_date - timedelta(days=30)  # Approximately 1 month

# Initialize a list to store minima for all tickers
minima_results = []
if not cleaned_data.empty:
    for ticker in cleaned_data.columns.get_level_values(0).unique():
        try:
            macd_series = cleaned_data[(ticker, 'MACD')].dropna()
            if len(macd_series) > 2:  # Need at least 3 points for minima
                # Find minima
                minima_dates = find_local_minima(macd_series)
                # Filter for minima within the past 1 month
                recent_minima = minima_dates[(minima_dates >= start_date) & (minima_dates <= end_date)]
                if not recent_minima.empty:
                    # Append rows with ticker, date, and MACD value
                    for date in recent_minima:
                        minima_results.append({
                            'Date': date,
                            'Ticker': ticker,
                            'MACD': macd_series.loc[date]
                        })
            else:
                print(f"{ticker}: Insufficient data for minima detection.")
        except Exception as e:
            print(f"Error finding minima for {ticker}: {e}")
else:
    print("No valid OHLCV or MACD data found for any ticker.")
    cleaned_data = pd.DataFrame()  # Ensure cleaned_data is an empty DataFrame if no valid data

# Step 7: Save local minima for the past 1 month to a readable CSV file on the Desktop
output_file = os.path.join(desktop_path, 'macd_minima.csv')
try:
    # Create DataFrame from minima results
    if minima_results:
        minima_df = pd.DataFrame(minima_results)
        # Sort by Date for readability
        minima_df = minima_df.sort_values(by='Date').reset_index(drop=True)
        # Format MACD to 4 decimal places for clarity
        minima_df['MACD'] = minima_df['MACD'].round(4)
        minima_df.to_csv(output_file, index=False)
        print(f"Local minima for the past 1 month saved to {output_file}")
    else:
        # Save an empty CSV with headers if no minima are found
        pd.DataFrame(columns=['Date', 'Ticker', 'MACD']).to_csv(output_file, index=False)
        print(f"No minima found for the past 1 month. Empty CSV saved to {output_file}")
except Exception as e:
    print(f"Error saving to {output_file}: {e}")
    # Try alternative file
    alt_output_file = os.path.join(desktop_path, 'macd_minima_backup.csv')
    try:
        minima_df.to_csv(alt_output_file, index=False) if minima_results else pd.DataFrame(columns=['Date', 'Ticker', 'MACD']).to_csv(alt_output_file, index=False)
        print(f"Local minima saved to alternative file: {alt_output_file}")
    except Exception as alt_e:
        print(f"Failed to save to {alt_output_file}: {alt_e}")
        raise ValueError("Unable to save MACD minima to CSV file.")

# The cleaned_data DataFrame is now available for further use
# Example: print(cleaned_data.shape) to check dimensions