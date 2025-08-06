import yfinance as yf
import talib
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime, timedelta
import os
import time

# New: Simple 2D Kalman Filter class for smoothing time series (position-velocity model)
class KalmanFilter:
    def __init__(self, F, H, Q, R, x0, P0):
        self.F = F  # State transition matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.x = x0  # Initial state
        self.P = P0  # Initial covariance

    def predict(self):
        # Predict step (no control input)
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        # Update step
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
        return self.x[0]  # Return smoothed position (MACD value)

# Function to smooth MACD using Kalman filter
def smooth_macd(macd_values, process_noise=0.1, measurement_noise=1.0):
    if len(macd_values) == 0:
        return np.array([])
    
    # Kalman parameters (tune these: higher process_noise reduces lag, lower reduces false positives)
    F = np.array([[1, 1], [0, 1]])  # State transition: position += velocity * dt (dt=1)
    H = np.array([[1, 0]])  # Observe position only
    Q = np.eye(2) * process_noise  # Process noise (controls responsiveness)
    R = np.array([[measurement_noise]])
    
    # Measurement noise (controls smoothing)
    x0 = np.array([[macd_values[0]], [0]])  # Initial state: position = first MACD, velocity=0
    P0 = np.eye(2)  # Initial covariance
    
    kf = KalmanFilter(F, H, Q, R, x0, P0)
    smoothed = []
    for val in macd_values:
        kf.predict()
        smoothed.append(kf.update(np.array([[val]])))
    return np.array(smoothed)

start_time = time.time()

# Read tickers from EQUITY_L.csv
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
equity_csv_path = os.path.join(desktop_path, "EQUITY_L.csv")
equity_df = pd.read_csv(equity_csv_path)
tickers = [symbol + ".NS" for symbol in equity_df['SYMBOL']]

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

all_trades = []
potential_dict = {}

for ticker in tickers:
    # Fetch market cap for the ticker
    stock = yf.Ticker(ticker)
    try:
        market_cap = stock.info.get('marketCap')
        if market_cap is None:
            print(f"No market cap data for {ticker}. Skipping.")
            continue
        # Convert market cap from USD to INR (1 crore = 10 million INR)
        # Assuming 1 USD = 83 INR (approximate as of 2025)
        market_cap_inr = market_cap * 83
        # Filter for market cap > 1000 crore (10 billion INR)
        if market_cap_inr < 1000 * 10**7:
            print(f"Market cap for {ticker} is {market_cap_inr/10**7:.2f} crore, below 1000 crore. Skipping.")
            continue
    except Exception as e:
        print(f"Error fetching market cap for {ticker}: {e}. Skipping.")
        continue

    data = yf.download(ticker, start='2000-01-01', interval='1wk')
    data = data.dropna()

    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(1, axis=1)

    close = data['Close'].to_numpy(dtype=np.float64)
    close = close.ravel()

    if len(close) < 35:
        print(f"Not enough data points to compute MACD for {ticker}. Skipping.")
        continue
    else:
        macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

    df = data.copy()
    df['MACD'] = macd
    df['Signal'] = signal
    df['Hist'] = hist

    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    df = df.dropna()

    macd_line_series = df['MACD']
    macd_line_values = macd_line_series.values

    # New: Smooth MACD using Kalman filter to reduce noise and lag
    smoothed_macd = smooth_macd(macd_line_values, process_noise=0.1, measurement_noise=1.0)
    
    # After computing smoothed_macd
    potential = True
    if len(smoothed_macd) >= 3:
        last = smoothed_macd[-1]
        prev1 = smoothed_macd[-2]
        prev2 = smoothed_macd[-3]
        if last < 0 and last < prev1 and last < prev2:
            potential = False
            print(f"Potential buy signal forming for {ticker} this week")

    potential_dict[ticker] = potential

    # Use smoothed MACD for extrema detection (reduce order to 2 for less lag post-smoothing)
    min_idx = argrelextrema(smoothed_macd, np.less, order=2)[0]
    max_idx = argrelextrema(smoothed_macd, np.greater, order=2)[0]

    buy_signals = set(idx for idx in min_idx if smoothed_macd[idx] < 0)
    sell_signals = set(idx for idx in max_idx if smoothed_macd[idx] > 0)

    trades = []
    position = False
    entry_date = None
    entry_price = None
    stop_loss = None

    for i in range(len(df)):
        date = df.index[i]
        close_price = df['Close'].iloc[i]  # Renamed to avoid conflict with 'close' keyword
        low = df['Low'].iloc[i]
        atr = df['ATR'].iloc[i]
        current_macd = df['MACD'].iloc[i]
        current_signal = df['Signal'].iloc[i]

        if position:
            # Update trailing stop-loss each week
            new_stop = close_price - (atr * 2.2)
            stop_loss = max(stop_loss, new_stop)

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
                continue  # Skip further checks if stopped out

            # Existing MACD peak sell signal
            if i in sell_signals:
                position = False
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

        if i in buy_signals and not position:
            position = True
            entry_date = date
            entry_price = close_price
            atr = df['ATR'].iloc[i]
            if np.isnan(atr):
                position = False
                continue
            stop_loss = entry_price - (atr * 2)  # Initial stop-loss

    if position:
        # Get the current close price (latest close from data)
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

    all_trades.extend(trades)
    if trades:
        print(f"Trades for {ticker}:")
        print(pd.DataFrame(trades))
    else:
        print(f"No trades for {ticker}.")

if all_trades:
    trades_df = pd.DataFrame(all_trades)
    trades_df['potential'] = trades_df['Ticker'].map(potential_dict)
    filename = f"all_trades_report_{timestamp}.csv"
    trades_df.to_csv(filename, index=False)
    print(f"Combined all trades report saved to '{filename}'")
else:
    print("No trades across all tickers.")

end_time = time.time()
total_time = end_time - start_time
print(f"Total time taken: {total_time:.2f} seconds")