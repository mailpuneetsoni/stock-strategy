import pandas as pd
import numpy as np
import os
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
DAILY_FILE    = os.path.join(desktop_path, 'stock_data_daily.csv')   # daily OHLCV
HOURLY_FILE   = os.path.join(desktop_path, 'stock_data.csv')          # hourly OHLCV
OUTPUT_FILE   = os.path.join(desktop_path, 'filtered_stocks.csv')

# -----------------------------
# LOAD DAILY CSV
# -----------------------------
print("Loading daily data...")
raw_daily = pd.read_csv(DAILY_FILE, header=[0, 1], index_col=0, parse_dates=True)
tickers = raw_daily.columns.get_level_values(0).unique().tolist()
print(f"Total tickers loaded: {len(tickers)}")

# Filter to last 1 year of daily data
one_year_ago = pd.Timestamp.today() - pd.DateOffset(years=1)
raw_daily    = raw_daily[raw_daily.index >= one_year_ago]
print(f"Daily data range: {raw_daily.index[0].strftime('%Y-%m-%d')} to {raw_daily.index[-1].strftime('%Y-%m-%d')}")

# -----------------------------
# GET SIGNAL DATE FROM DAILY CSV
# -----------------------------
sample_close = raw_daily[tickers[0]]['Close'].dropna()
scan_date    = sample_close.index[-1]                          # Timestamp, not string
scan_date_str = scan_date.strftime('%Y-%m-%d')
print(f"Signal date (from daily CSV): {scan_date_str}")

# -----------------------------
# LOAD HOURLY CSV
# -----------------------------
print("\nLoading hourly data...")
raw_hourly = pd.read_csv(HOURLY_FILE, header=[0, 1], index_col=0, parse_dates=True)
hourly_tickers = raw_hourly.columns.get_level_values(0).unique().tolist()
print(f"Hourly tickers loaded: {len(hourly_tickers)}")

# -----------------------------
# HELPER: LAST MONTH'S FINAL CLOSE
# Derived from scan_date — not today's clock
# -----------------------------
def get_last_month_close(ticker, raw_daily, scan_date):
    """
    Returns the closing price on the last trading day of the month
    BEFORE the scan_date month.
    """
    try:
        close = raw_daily[ticker]['Close'].dropna()

        # First day of the scan_date's month
        first_of_scan_month = scan_date.replace(day=1)

        # All data strictly before that boundary = previous month and earlier
        prev_month_data = close[close.index < first_of_scan_month]

        if prev_month_data.empty:
            return None

        return prev_month_data.iloc[-1]   # Last trading day of previous month

    except Exception as e:
        print(f"  [{ticker}] last_month_close error: {e}")
        return None

# -----------------------------
# HELPER: RULE 1 — Z-SCORE CHECK (daily data)
# -----------------------------
def check_rule1(ticker, raw_daily, window=200, threshold=-1.4):
    """
    Z-score of daily close vs 200-day SMA <= -1.4
    Uses only daily CSV data — no live fetch.
    """
    try:
        close = raw_daily[ticker]['Close'].dropna()

        if len(close) < window:
            return False, None

        sma    = close.rolling(window).mean()
        std    = close.rolling(window).std()
        zscore = (close - sma) / std

        latest_zscore = zscore.iloc[-1]
        return latest_zscore <= threshold, round(float(latest_zscore), 4)

    except Exception as e:
        print(f"  [{ticker}] Rule1 error: {e}")
        return False, None

# -----------------------------
# HELPER: RULE 2 — 100-HOUR EMA vs LAST MONTH CLOSE (hourly CSV)
# No live API call — reads from hourly CSV instead
# -----------------------------
def check_rule2(ticker, last_month_close, raw_hourly, scan_date,
                ema_period=100, threshold=0.90):
    """
    100-hour EMA of hourly close < 90% of last month's final close.
    Reads from the hourly CSV — uses only data up to scan_date.
    """
    try:
        if last_month_close is None or last_month_close == 0:
            return False, None, None

        if ticker not in raw_hourly.columns.get_level_values(0):
            print(f"  [{ticker}] Not found in hourly CSV — skipping Rule 2")
            return False, None, None

        # Use only hourly data up to and including the scan date
        hourly_close = raw_hourly[ticker]['Close'].dropna()
        hourly_close = hourly_close[hourly_close.index.normalize() <= scan_date]

        if len(hourly_close) < ema_period:
            print(f"  [{ticker}] Not enough hourly bars (got {len(hourly_close)}, need {ema_period})")
            return False, None, None

        ema_100    = hourly_close.ewm(span=ema_period, adjust=False).mean()
        latest_ema = ema_100.iloc[-1]
        target     = threshold * last_month_close

        return latest_ema < target, round(float(latest_ema), 2), round(float(target), 2)

    except Exception as e:
        print(f"  [{ticker}] Rule2 error: {e}")
        return False, None, None

# -----------------------------
# MAIN SCREENING LOOP
# -----------------------------
results = []

print(f"\n{'='*60}")
print(f"Screening {len(tickers)} tickers for date: {scan_date_str}")
print(f"{'='*60}\n")

for i, ticker in enumerate(tickers):
    print(f"[{i+1}/{len(tickers)}] Checking {ticker}...")

    # --- Rule 1: Z-score ---
    r1_pass, zscore = check_rule1(ticker, raw_daily)
    if not r1_pass:
        print(f"  FAIL Rule1 | Z-score: {zscore}")
        continue
    print(f"  PASS Rule1 | Z-score: {zscore}")

    # --- Last month close (anchor for Rule 2) ---
    last_month_close = get_last_month_close(ticker, raw_daily, scan_date)
    if last_month_close is None:
        print(f"  SKIP — could not get last month close for {scan_date_str}")
        continue

    # --- Rule 2: 100h EMA vs last month close ---
    r2_pass, ema_val, target_val = check_rule2(
        ticker, last_month_close, raw_hourly, scan_date
    )
    if not r2_pass:
        print(f"  FAIL Rule2 | 100h EMA: {ema_val}, Target (<90% of {round(last_month_close, 2)}): {target_val}")
        continue
    print(f"  PASS Rule2 | 100h EMA: {ema_val} < {target_val} (90% of {round(last_month_close, 2)})")

    print(f"  ✅ {ticker} passes ALL rules")
    results.append({
        'Date'            : scan_date_str,
        'Ticker'          : ticker,
        'Z_Score'         : zscore,
        'Last_Month_Close': round(float(last_month_close), 2),
        'EMA_100h'        : ema_val,
        'Target_90pct'    : target_val,
    })

# -----------------------------
# SAVE OUTPUT — append with duplicate guard
# -----------------------------
print(f"\n{'='*60}")

if results:
    df_new = pd.DataFrame(results)

    if os.path.exists(OUTPUT_FILE):
        df_existing = pd.read_csv(OUTPUT_FILE)

        # Remove any existing rows for the same date to prevent duplicates
        df_existing = df_existing[df_existing['Date'] != scan_date_str]

        df_out = pd.concat([df_existing, df_new], ignore_index=True)
        df_out.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ {len(results)} signals saved for {scan_date_str} → {OUTPUT_FILE}")
    else:
        df_new.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ {len(results)} signals saved (new file) → {OUTPUT_FILE}")

    print(f"\n{df_new.to_string(index=False)}")

else:
    print(f"⚠️  No stocks passed all filters on {scan_date_str}.")