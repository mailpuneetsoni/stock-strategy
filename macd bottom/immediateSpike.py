import pandas as pd
import numpy as np
import os
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
desktop_path  = os.path.join(os.path.expanduser('~'), 'Desktop')
DAILY_FILE    = os.path.join(desktop_path, 'stock_data_daily.csv')   # daily OHLCV
HOURLY_FILE   = os.path.join(desktop_path, 'stock_data.csv')          # hourly OHLCV
OUTPUT_FILE   = os.path.join(desktop_path, 'backtest_signals.csv')    # backtest output

# How many past trading days to backtest over
LOOKBACK_DAYS = 365

# Rule parameters
ZSCORE_WINDOW    = 200       # days for SMA and STD
ZSCORE_THRESHOLD = -1.4      # Rule 1: Z-score must be <= this
EMA_PERIOD       = 100       # hourly bars for EMA
EMA_THRESHOLD    = 0.90      # Rule 2: EMA must be < 90% of last month close

# -----------------------------
# LOAD DAILY CSV
# -----------------------------
print("Loading daily data...")
raw_daily = pd.read_csv(DAILY_FILE, header=[0, 1], index_col=0, parse_dates=True)
raw_daily.index = pd.to_datetime(raw_daily.index).tz_localize(None)   # strip timezone
tickers   = raw_daily.columns.get_level_values(0).unique().tolist()
print(f"Tickers loaded: {len(tickers)}")
print(f"Daily data range: {raw_daily.index[0].strftime('%Y-%m-%d')} to {raw_daily.index[-1].strftime('%Y-%m-%d')}")

# -----------------------------
# LOAD HOURLY CSV
# -----------------------------
print("\nLoading hourly data...")
raw_hourly = pd.read_csv(HOURLY_FILE, header=[0, 1], index_col=0, parse_dates=True)
raw_hourly.index = pd.to_datetime(raw_hourly.index).tz_localize(None)  # strip timezone
hourly_tickers = raw_hourly.columns.get_level_values(0).unique().tolist()
print(f"Hourly tickers loaded: {len(hourly_tickers)}")

# -----------------------------
# BUILD LIST OF BACKTEST DATES
# Get all trading days in the daily CSV for the past LOOKBACK_DAYS
# But we need at least ZSCORE_WINDOW days of history before each date
# so we need the CSV to go back far enough
# -----------------------------
all_trading_days = raw_daily.index.sort_values()

# Only backtest dates that have at least ZSCORE_WINDOW bars of history before them
min_history_date = all_trading_days[ZSCORE_WINDOW]   # need 200 bars before this date

# Backtest window: last LOOKBACK_DAYS trading days that have enough history
cutoff_date = all_trading_days[-1] - pd.DateOffset(days=LOOKBACK_DAYS)
backtest_dates = all_trading_days[
    (all_trading_days >= max(cutoff_date, min_history_date))
]

print(f"\nBacktest period: {backtest_dates[0].strftime('%Y-%m-%d')} to {backtest_dates[-1].strftime('%Y-%m-%d')}")
print(f"Total trading days to scan: {len(backtest_dates)}")

# -----------------------------
# HELPER: LAST MONTH CLOSE
# For a given scan_date, return the last close of the previous month
# -----------------------------
def get_last_month_close(ticker, raw_daily, scan_date):
    try:
        close = raw_daily[ticker]['Close'].dropna()
        first_of_scan_month = scan_date.replace(day=1)
        prev_month_data = close[close.index < first_of_scan_month]
        prev_month_data = close[close.index < first_of_scan_month]
        if prev_month_data.empty:
            return None

        last_close_date = prev_month_data.index[-1]
        last_close = float(prev_month_data.iloc[-1])
        print(f"{ticker} last month close for {last_close_date.strftime('%Y-%m')} = {last_close:.2f}")
        return last_close
        if prev_month_data.empty:
            return None
        return float(prev_month_data.iloc[-1])
    except:
        return None

# -----------------------------
# HELPER: RULE 1 — Z-SCORE
# Uses only daily data UP TO and INCLUDING scan_date
# -----------------------------
def check_rule1(ticker, raw_daily, scan_date, window=200, threshold=-1.4):
    try:
        close = raw_daily[ticker]['Close'].dropna()
        # Only data up to scan_date — no future leakage
        close = close[close.index <= scan_date]

        if len(close) < window:
            return False, None

        sma    = close.rolling(window).mean()
        std    = close.rolling(window).std()
        zscore = (close - sma) / std

        latest_zscore = float(zscore.iloc[-1])
        return latest_zscore <= threshold, round(latest_zscore, 4)
    except:
        return False, None

# -----------------------------
# HELPER: RULE 2 — 100-HOUR EMA
# Uses only hourly data UP TO and INCLUDING scan_date
# -----------------------------
def check_rule2(ticker, last_month_close, raw_hourly, scan_date,
                ema_period=100, threshold=0.90):
    try:
        if last_month_close is None or last_month_close == 0:
            return False, None, None

        if ticker not in raw_hourly.columns.get_level_values(0):
            return False, None, None

        hourly_close = raw_hourly[ticker]['Close'].dropna()
        # Only hourly bars up to end of scan_date — no future leakage
        hourly_close = hourly_close[hourly_close.index.normalize() <= scan_date]

        if len(hourly_close) < ema_period:
            return False, None, None

        ema_100    = hourly_close.ewm(span=ema_period, adjust=False).mean()
        latest_ema = float(ema_100.iloc[-1])
        target     = threshold * last_month_close

        return latest_ema < target, round(latest_ema, 2), round(target, 2)
    except:
        return False, None, None

# -----------------------------
# BACKTEST LOOP
# For each trading date, run both rules on all tickers
# -----------------------------
all_signals = []
total_dates = len(backtest_dates)

for day_num, scan_date in enumerate(backtest_dates):
    scan_date_str = scan_date.strftime('%Y-%m-%d')
    day_signals   = 0

    print(f"\n[Day {day_num+1}/{total_dates}] Scanning {scan_date_str}...")

    for ticker in tickers:

        # Rule 1
        r1_pass, zscore = check_rule1(ticker, raw_daily, scan_date)
        if not r1_pass:
            continue

        # Last month close
        last_month_close = get_last_month_close(ticker, raw_daily, scan_date)
        if last_month_close is None:
            continue

        # Rule 2
        r2_pass, ema_val, target_val = check_rule2(
            ticker, last_month_close, raw_hourly, scan_date
        )
        if not r2_pass:
            continue

        # Both rules passed
        all_signals.append({
            'Date'            : scan_date_str,
            'Ticker'          : ticker,
            'Z_Score'         : zscore,
            'Last_Month_Close': round(last_month_close, 2),
            'EMA_100h'        : ema_val,
            'Target_90pct'    : target_val,
        })
        day_signals += 1

    print(f"  → {day_signals} signal(s) on {scan_date_str}")

# -----------------------------
# SAVE BACKTEST OUTPUT
# -----------------------------
print(f"\n{'='*60}")
if all_signals:
    df_out = pd.DataFrame(all_signals)
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Backtest complete.")
    print(f"   Total signals: {len(all_signals)}")
    print(f"   Date range   : {df_out['Date'].min()} to {df_out['Date'].max()}")
    print(f"   Unique tickers: {df_out['Ticker'].nunique()}")
    print(f"   Saved to     : {OUTPUT_FILE}")
    print(f"\nSignals per date (top 20):")
    print(df_out.groupby('Date').size().rename('Signals').tail(20).to_string())
else:
    print("⚠️  No signals found across the entire backtest period.")
    print("   Check that your CSV files have enough data and the right date ranges.")x