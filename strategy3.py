import yfinance as yf
import talib
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime, timedelta

tickers = ['ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BEL.NS', 'BHARTIARTL.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'ZOMATO.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'INFY.NS', 'ITC.NS', 'JIOFIN.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SHRIRAMFIN.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'TRENT.NS', 'ULTRACEMCO.NS', 'WIPRO.NS']

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

all_trades = []

for ticker in tickers:
    # Download historical daily data
    data = yf.download(ticker, start='2000-01-01', interval='1wk')
    data = data.dropna()

    # Handle multi-index columns issue in recent yfinance versions for single tickers
    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(1, axis=1)

    close = data['Close'].to_numpy(dtype=np.float64)
    close = close.ravel()

    if len(close) < 35:
        print(f"Not enough data points to compute MACD for {ticker}. Skipping.")
        continue
    else:
        # Compute MACD using TA-Lib
        macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

    # Add to DataFrame
    df = data.copy()
    df['MACD'] = macd
    df['Signal'] = signal
    df['Hist'] = hist
    df = df.dropna()

    # Find local minima in macd line where < 0 for buy, local maxima > 0 for sell
    macd_line_series = df['MACD']
    macd_line_values = macd_line_series.values

    # Use order=15 for detecting extrema 
    min_idx = argrelextrema(macd_line_values, np.less, order=3)[0]
    max_idx = argrelextrema(macd_line_values, np.greater, order=3)[0]

    buy_signals = [idx for idx in min_idx if macd_line_values[idx] < 0]
    sell_signals = [idx for idx in max_idx if macd_line_values[idx] > 0]

    # Combine and sort signals
    all_signals = [(idx, 'buy') for idx in buy_signals] + [(idx, 'sell') for idx in sell_signals]
    all_signals.sort(key=lambda x: x[0])

    # Simulate trades (long only, buy on bottom, sell on next top)
    trades = []
    position = False
    entry_date = None
    entry_price = None

    for idx, sig_type in all_signals:
        date = df.index[idx]
        price = df['Close'].iloc[idx]
        if sig_type == 'buy' and not position:
            position = True
            entry_date = date
            entry_price = price
        elif sig_type == 'sell' and position:
            position = False
            exit_date = date
            exit_price = price
            ret = (exit_price - entry_price) / entry_price * 100
            trades.append({
                'Ticker': ticker,
                'Entry Date': entry_date.strftime('%Y-%m-%d'),
                'Entry Price': round(entry_price, 2),
                'Exit Date': exit_date.strftime('%Y-%m-%d'),
                'Exit Price': round(exit_price, 2),
                'Return %': round(ret, 2)
            })

    if trades:
        all_trades.extend(trades)
        print(f"Trades for {ticker}:")
        print(pd.DataFrame(trades))
    else:
        print(f"No trades generated in the backtest for {ticker}.")

# Save all trades to a single CSV
if all_trades:
    trades_df = pd.DataFrame(all_trades)
    filename = f"backtest_report_all_{timestamp}.csv"
    trades_df.to_csv(filename, index=False)
    print(f"Combined backtest report saved to '{filename}'")
else:
    print("No trades generated across all tickers.")