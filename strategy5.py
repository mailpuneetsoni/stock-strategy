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

# Read tickers from EQUITY_L.csv
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
equity_csv_path = os.path.join(desktop_path, "EQUITY_L.csv")
equity_df = pd.read_csv(equity_csv_path)
tickers = [symbol + ".NS" for symbol in equity_df['SYMBOL']]

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

all_trades = []
potential_dict = {}

for ticker in tickers:
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
    potential = False
    if len(smoothed_macd) >= 3:
        last = smoothed_macd[-1]
        prev1 = smoothed_macd[-2]
        prev2 = smoothed_macd[-3]
        if last < 0 and last < prev1 and last < prev2:
            potential = True
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
        close = df['Close'].iloc[i]
        low = df['Low'].iloc[i]

        if position:
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

        if i in buy_signals and not position:
            position = True
            entry_date = date
            entry_price = close
            atr = df['ATR'].iloc[i]
            if np.isnan(atr):
                position = False
                continue
            stop_loss = entry_price - (atr * 2)

        elif i in sell_signals and position:
            position = False
            exit_date = date
            exit_price = close
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

    if position:
        trades.append({
            'Ticker': ticker,
            'Entry Date': entry_date.strftime('%Y-%m-%d'),
            'Entry Price': round(entry_price, 2),
            'Exit Date': None,
            'Exit Price': None,
            'Return %': None,
            'Status': 'Open'
        })

    if trades:
        all_trades.extend(trades)
        print(f"Trades for {ticker}:")
        print(pd.DataFrame(trades))
    else:
        print(f"No trades generated in the backtest for {ticker}.")

if all_trades:
    trades_df = pd.DataFrame(all_trades)
    trades_df['potential'] = trades_df['Ticker'].map(potential_dict)
    filename = f"backtest_report_all_{timestamp}.csv"
    trades_df.to_csv(filename, index=False)
    print(f"Combined backtest report saved to '{filename}'")
else:
    print("No trades generated across all tickers.")
