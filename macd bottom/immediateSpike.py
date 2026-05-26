import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime, timedelta

# -----------------------------
# CONFIG
# -----------------------------
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
CSV_FILE     = os.path.join(desktop_path, 'stock_data.csv')
OUTPUT_FILE  = os.path.join(desktop_path, 'filtered_stocks.csv')

# -----------------------------
# LOAD DAILY CSV
# -----------------------------
raw = pd.read_csv(CSV_FILE, header=[0, 1], index_col=0, parse_dates=True)
tickers = raw.columns.get_level_values(0).unique().tolist()
print(f"Total tickers loaded: {len(tickers)}")

# -----------------------------
# GET SIGNAL DATE FROM CSV  ← now correctly placed AFTER raw is loaded
# -----------------------------
sample_close = raw[tickers[0]]['Close'].dropna()
scan_date = sample_close.index[-1].strftime('%Y-%m-%d')
print(f"Signal date (from stock_data.csv): {scan_date}")

# -----------------------------
# HELPER: LAST MONTH'S FINAL CLOSE (from daily data)
# -----------------------------
def get_last_month_close(ticker, raw):
    """Returns the closing price on the last trading day of the previous month."""
    try:
        close = raw[ticker]['Close'].dropna()
        today = pd.Timestamp.today()
        first_of_this_month = today.replace(day=1)
        prev_month_data = close[close.index < first_of_this_month]
        if prev_month_data.empty:
            return None
        return prev_month_data.iloc[-1]
    except Exception as e:
        print(f"  [{ticker}] last_month_close error: {e}")
        return None

# -----------------------------
# HELPER: RULE 1 — Z-SCORE CHECK (daily data)
# -----------------------------
def check_rule1(ticker, raw, window=200, threshold=-1.4):
    """Z-score of daily close vs 200-day SMA <= -1.4"""
    try:
        close = raw[ticker]['Close'].dropna()
        if len(close) < window:
            return False, None
        sma    = close.rolling(window).mean()
        std    = close.rolling(window).std()
        zscore = (close - sma) / std
        latest_zscore = zscore.iloc[-1]
        return latest_zscore <= threshold, round(latest_zscore, 4)
    except Exception as e:
        print(f"  [{ticker}] Rule1 error: {e}")
        return False, None

# -----------------------------
# HELPER: RULE 2 — 100-HOUR EMA vs LAST MONTH CLOSE (hourly data)
# -----------------------------
def check_rule2(ticker, last_month_close, ema_period=100, threshold=0.90):
    """100-hour EMA of hourly close < 90% of last month's final close."""
    try:
        if last_month_close is None or last_month_close == 0:
            return False, None, None

        end    = datetime.today()
        start  = end - timedelta(days=14)
        hourly = yf.download(ticker, start=start, end=end, interval='1h',
                             progress=False, auto_adjust=True)

        if hourly.empty or len(hourly) < ema_period:
            print(f"  [{ticker}] Not enough hourly data (got {len(hourly)} bars)")
            return False, None, None

        ema_100    = hourly['Close'].ewm(span=ema_period, adjust=False).mean()
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

for i, ticker in enumerate(tickers):
    print(f"[{i+1}/{len(tickers)}] Checking {ticker}...")

    # --- Rule 1 ---
    r1_pass, zscore = check_rule1(ticker, raw)
    if not r1_pass:
        print(f"  FAIL Rule1 | Z-score: {zscore}")
        continue
    print(f"  PASS Rule1 | Z-score: {zscore}")

    # --- Last month close ---
    last_month_close = get_last_month_close(ticker, raw)
    if last_month_close is None:
        print(f"  SKIP — could not get last month close")
        continue

    # --- Rule 2 ---
    r2_pass, ema_val, target_val = check_rule2(ticker, last_month_close)
    if not r2_pass:
        print(f"  FAIL Rule2 | 100h EMA: {ema_val}, Target (<90% of {round(last_month_close,2)}): {target_val}")
        continue
    print(f"  PASS Rule2 | 100h EMA: {ema_val} < {target_val} (90% of {round(last_month_close,2)})")

    print(f"  ✅ {ticker} passes ALL rules")
    results.append({
        'Date'            : scan_date,
        'Ticker'          : ticker,
        'Z_Score'         : zscore,
        'Last_Month_Close': round(last_month_close, 2),
        'EMA_100h'        : ema_val,
        'Target_90pct'    : target_val,
    })

# -----------------------------
# SAVE OUTPUT — append mode for backtesting
# -----------------------------
if results:
    df_out = pd.DataFrame(results)
    if os.path.exists(OUTPUT_FILE):
        df_out.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
        print(f"\n✅ {len(results)} stocks appended to: {OUTPUT_FILE}")
    else:
        df_out.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ {len(results)} stocks saved to: {OUTPUT_FILE}")
    print(df_out.to_string(index=False))
else:
    print(f"\n⚠️ No stocks passed all filters on {scan_date}.")