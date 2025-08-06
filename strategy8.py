import yfinance as yf
import talib
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime, timedelta
import os
import time

# --- Kalman Filter Implementation ---

class KalmanFilter:
    """A 2D Kalman Filter for smoothing time series data (position-velocity model)."""
    def __init__(self, F, H, Q, R, x0, P0):
        """
        Initialize the Kalman Filter.
        
        Args:
            F: State transition matrix (numpy array).
            H: Observation matrix (numpy array).
            Q: Process noise covariance matrix (numpy array).
            R: Measurement noise covariance matrix (numpy array).
            x0: Initial state vector (numpy array).
            P0: Initial covariance matrix (numpy array).
        """
        self.F = F  # State transition matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.x = x0  # Initial state
        self.P = P0  # Initial covariance

    def predict(self):
        """Predict the next state without new measurements."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        """
        Update the state with a new measurement.
        
        Args:
            z: Measurement vector (numpy array).
        
        Returns:
            float: Smoothed position (MACD value).
        """
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
        return self.x[0]  # Return smoothed position (MACD value)

def smooth_macd(macd_values, process_noise=0.1, measurement_noise=1.0):
    """
    Smooth MACD values using a Kalman Filter to reduce noise and lag.
    
    Args:
        macd_values: Array of MACD values (numpy array).
        process_noise: Process noise level (float, default 0.1).
        measurement_noise: Measurement noise level (float, default 1.0).
    
    Returns:
        numpy array: Smoothed MACD values.
    """
    if len(macd_values) == 0:
        return np.array([])
    
    # Kalman parameters
    F = np.array([[1, 1], [0, 1]])  # State transition: position += velocity * dt (dt=1)
    H = np.array([[1, 0]])  # Observe position only
    Q = np.eye(2) * process_noise  # Process noise (controls responsiveness)
    R = np.array([[measurement_noise]])  # Measurement noise (controls smoothing)
    x0 = np.array([[macd_values[0]], [0]])  # Initial state: position = first MACD, velocity=0
    P0 = np.eye(2)  # Initial covariance
    
    kf = KalmanFilter(F, H, Q, R, x0, P0)
    smoothed = []
    for val in macd_values:
        kf.predict()
        smoothed.append(kf.update(np.array([[val]])))
    return np.array(smoothed)

# --- Main Script ---

def load_tickers():
    """
    Load ticker symbols from EQUITY_L.csv.
    
    Returns:
        list: List of ticker symbols with '.NS' suffix.
    """
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    equity_csv_path = os.path.join(desktop_path, "EQUITY_L.csv")
    equity_df = pd.read_csv(equity_csv_path)
    return [symbol + ".NS" for symbol in equity_df['SYMBOL']]

def filter_tickers_by_market_cap(tickers):
    """
    Filter tickers based on market cap (> 1000 crore INR).
    
    Args:
        tickers: List of ticker symbols.
    
    Returns:
        list: Filtered list of tickers meeting market cap criteria.
    """
    filtered_tickers = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        try:
            market_cap = stock.info.get('marketCap')
            if market_cap is None:
                print(f"No market cap data for {ticker}. Skipping.")
                continue
            # Convert market cap from USD to INR (1 crore = 10 million INR)
            market_cap_inr = market_cap * 83  # Assuming 1 USD = 83 INR (2025 estimate)
            if market_cap_inr >= 1000 * 10**7:
                filtered_tickers.append(ticker)
            else:
                print(f"Market cap for {ticker} is {market_cap_inr/10**7:.2f} crore, below 1000 crore. Skipping.")
        except Exception as e:
            print(f"Error fetching market cap for {ticker}: {e}. Skipping.")
            continue
    return filtered_tickers

def download_ticker_data(tickers):
    """
    Download historical weekly data for all tickers.
    
    Args:
        tickers: List of ticker symbols.
    
    Returns:
        pandas DataFrame: Historical data for all tickers, or empty DataFrame if download fails.
    """
    if tickers:
        try:
            data = yf.download(tickers, start='2000-01-01', interval='1wk', auto_adjust=False)
            print(f"Downloaded data for {len(tickers)} tickers. Columns: {data.columns.tolist()}")
            return data
        except Exception as e:
            print(f"Error downloading data for tickers: {e}. Exiting.")
            return pd.DataFrame()
    return pd.DataFrame()

def process_ticker_data(ticker, all_data, potential_dict, all_trades):
    """
    Process data for a single ticker, compute indicators, and simulate trades.
    
    Args:
        ticker: Ticker symbol (str).
        all_data: DataFrame containing historical data for all tickers.
        potential_dict: Dictionary to store potential buy signals.
        all_trades: List to store trade records.
    """
    # Check if data exists for the ticker
    if all_data.empty:
        print(f"No data available for any tickers. Skipping {ticker}.")
        return
    
    # Handle multi-level or single-level columns
    try:
        if all_data.columns.nlevels > 1:
            if ticker not in all_data.columns.get_level_values(1):
                print(f"No data downloaded for {ticker}. Skipping.")
                return
            ticker_data = all_data.xs(ticker, level=1, axis=1).dropna()
        else:
            if ticker not in all_data.columns:
                print(f"No data downloaded for {ticker}. Skipping.")
                return
            ticker_data = all_data[ticker].dropna()
    except Exception as e:
        print(f"Error accessing data for {ticker}: {e}. Skipping.")
        return

    # Ensure sufficient data for MACD calculation
    close = ticker_data['Close'].to_numpy(dtype=np.float64)
    close = close.ravel()
    if len(close) < 35:
        print(f"Not enough data points to compute MACD for {ticker}. Skipping.")
        return

    # Compute MACD and ATR
    macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df = ticker_data.copy()
    df['MACD'] = macd
    df['Signal'] = signal
    df['Hist'] = hist
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df = df.dropna()

    # Smooth MACD
    macd_line_values = df['MACD'].values
    smoothed_macd = smooth_macd(macd_line_values, process_noise=0.1, measurement_noise=1.0)
    
    # Check for potential buy signal
    potential = True
    if len(smoothed_macd) >= 3:
        last = smoothed_macd[-1]
        prev1 = smoothed_macd[-2]
        prev2 = smoothed_macd[-3]
        if last < 0 and last < prev1 and last < prev2:
            potential = False
            print(f"Potential buy signal forming for {ticker} this week")
    potential_dict[ticker] = potential

    # Detect buy/sell signals using smoothed MACD
    min_idx = argrelextrema(smoothed_macd, np.less, order=2)[0]
    max_idx = argrelextrema(smoothed_macd, np.greater, order=2)[0]
    buy_signals = set(idx for idx in min_idx if smoothed_macd[idx] < 0)
    sell_signals = set(idx for idx in max_idx if smoothed_macd[idx] > 0)

    # Simulate trades
    trades = []
    position = False
    entry_date = None
    entry_price = None
    stop_loss = None

    for i in range(len(df)):
        date = df.index[i]
        close_price = df['Close'].iloc[i]
        low = df['Low'].iloc[i]
        atr = df['ATR'].iloc[i]
        current_macd = df['MACD'].iloc[i]
        current_signal = df['Signal'].iloc[i]

        if position:
            # Update trailing stop-loss
            new_stop = close_price - (atr * 2.2)
            stop_loss = max(stop_loss, new_stop)

            # Exit if low price hits stop-loss
            if low <= stop_loss:
                exit_date = date
                exit_price = stop_loss
                ret = (exit_price - entry_price) / entry_price * 100
                trades.append({
                    'Ticker': ticker,
                    'Entry Date': entry_date.strftime('%Y-%m-%d'),
                    'Entry Price': round(entry_price, 2),
                    'Exit Date': exit_date.strftime('%Y-%m-%d'),
                    'Exit Price': round(exit_price, 2),
                    'Return %': round(ret, 2),
                    'Status': 'Closed'
                })
                position = False
                continue

            # Exit on MACD peak sell signal
            if i in sell_signals:
                exit_date = date
                exit_price = close_price
                ret = (exit_price - entry_price) / entry_price * 100
                trades.append({
                    'Ticker': ticker,
                    'Entry Date': entry_date.strftime('%Y-%m-%d'),
                    'Entry Price': round(entry_price, 2),
                    'Exit Date': exit_date.strftime('%Y-%m-%d'),
                    'Exit Price': round(exit_price, 2),
                    'Return %': round(ret, 2),
                    'Status': 'Closed'
                })
                position = False
                continue

        # Enter position on buy signal
        if i in buy_signals and not position:
            position = True
            entry_date = date
            entry_price = close_price
            atr = df['ATR'].iloc[i]
            if np.isnan(atr):
                position = False
                continue
            stop_loss = entry_price - (atr * 2)  # Initial stop-loss

    # Record open position if any
    if position:
        current_close = df['Close'].iloc[-1]
        current_return = ((current_close - entry_price) / entry_price) * 100
        trades.append({
            'Ticker': ticker,
            'Entry Date': entry_date.strftime('%Y-%m-%d'),
            'Entry Price': round(entry_price, 2),
            'Exit Date': None,
            'Exit Price': None,
            'Return %': round(current_return, 2),
            'Status': 'Open'
        })

    # Log trades
    if trades:
        print(f"Trades for {ticker}:")
        print(pd.DataFrame(trades))
    else:
        print(f"No trades for {ticker}.")
    
    all_trades.extend(trades)

def main():
    """Main function to orchestrate the trading strategy execution."""
    start_time = time.time()
    
    # Initialize storage
    all_trades = []
    potential_dict = {}
    
    # Load and filter tickers
    tickers = load_tickers()
    filtered_tickers = filter_tickers_by_market_cap(tickers)
    
    # Download historical data
    all_data = download_ticker_data(filtered_tickers)
    
    # Process each ticker
    for ticker in filtered_tickers:
        process_ticker_data(ticker, all_data, potential_dict, all_trades)
    
    # Save trade report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df['potential'] = trades_df['Ticker'].map(potential_dict)
        filename = f"all_trades_report_{timestamp}.csv"
        trades_df.to_csv(filename, index=False)
        print(f"Combined all trades report saved to '{filename}'")
    else:
        print("No trades across all tickers.")
    
    # Log execution time
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()