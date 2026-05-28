import pandas as pd
import numpy as np
import os

# -----------------------------
# CONFIG
# -----------------------------
desktop_path  = os.path.join(os.path.expanduser('~'), 'Desktop')
DAILY_FILE    = os.path.join(desktop_path, 'stock_data_daily.csv')   # daily OHLCV
HOURLY_FILE   = os.path.join(desktop_path, 'stock_data.csv')          # hourly OHLCV
OUTPUT_FILE   = os.path.join(desktop_path, 'backtest_signals.csv')    # backtest output
LOOKBACK_DAYS = 365
ZSCORE_WINDOW    = 200
ZSCORE_THRESHOLD = -1.4
EMA_PERIOD       = 100
EMA_THRESHOLD    = 0.90

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
print(f"Hourly data range  : {raw_hourly.index[0].strftime('%Y-%m-%d')} to {raw_hourly.index[-1].strftime('%Y-%m-%d')}")

# -----------------------------
# PRE-COMPUTE: Z-SCORES for all tickers on all daily dates
# This avoids recomputing Z-score inside the hourly loop
# Z-score on a given day applies to ALL hourly bars within that day
# -----------------------------
print("\nPre-computing Z-scores for all tickers and dates...")

zscore_cache = {}   # {ticker: pd.Series indexed by date}

for ticker in tickers:
    try:
        close = raw_daily[ticker]['Close'].dropna()
        sma   = close.rolling(ZSCORE_WINDOW).mean()
        std   = close.rolling(ZSCORE_WINDOW).std()
        z     = (close - sma) / std
        zscore_cache[ticker] = z
    except:
        zscore_cache[ticker] = pd.Series(dtype=float)

print("Z-score pre-computation complete.")

# -----------------------------
# PRE-COMPUTE: LAST MONTH CLOSE for all tickers on all daily dates
# For each daily date, what was the last close of the previous month?
# -----------------------------
print("Pre-computing last month closes...")

last_month_close_cache = {}   # {ticker: pd.Series indexed by date}

for ticker in tickers:
    try:
        close        = raw_daily[ticker]['Close'].dropna()
        monthly_last = close.resample('ME').last()   # last close of each month

        # For each daily date, look up the previous month's last close
        result = {}
        for date in close.index:
            # Previous month end = one month before current month start
            prev_month_end = (date.replace(day=1) - pd.DateOffset(days=1))
            prev_month_key = prev_month_end.to_period('M').to_timestamp('M')

            # Find the closest available month-end close on or before prev_month_key
            available = monthly_last[monthly_last.index <= prev_month_key]
            if not available.empty:
                result[date] = float(available.iloc[-1])
            else:
                result[date] = None

        last_month_close_cache[ticker] = result
    except:
        last_month_close_cache[ticker] = {}

print("Last month close pre-computation complete.")

# -----------------------------
# BUILD LIST OF HOURLY BARS TO SCAN
# Only hourly bars within the backtest window
# -----------------------------
all_trading_days = raw_daily.index.sort_values()
min_history_date = all_trading_days[ZSCORE_WINDOW] if len(all_trading_days) > ZSCORE_WINDOW else all_trading_days[0]
cutoff_date      = all_trading_days[-1] - pd.DateOffset(days=LOOKBACK_DAYS)
start_date       = max(cutoff_date, min_history_date)

# All hourly timestamps within the backtest window
hourly_timestamps = raw_hourly.index.sort_values()
backtest_hourly   = hourly_timestamps[hourly_timestamps >= start_date]

print(f"\nBacktest period    : {start_date.strftime('%Y-%m-%d')} to {all_trading_days[-1].strftime('%Y-%m-%d')}")
print(f"Total hourly bars  : {len(backtest_hourly)}")
print(f"Total tickers      : {len(tickers)}")
print(f"Estimated iterations: {len(backtest_hourly) * len(tickers):,}")

# -----------------------------
# HOURLY BACKTEST LOOP
# For each hourly timestamp, check both rules for all tickers
# -----------------------------
all_signals  = []
total_hours  = len(backtest_hourly)
last_printed = -1

print("\nStarting hourly backtest scan...\n")

for hour_num, hourly_ts in enumerate(backtest_hourly):

    # Progress print every 100 hours
    pct = int(hour_num * 100 / total_hours)
    if pct % 5 == 0 and pct != last_printed:
        print(f"  Progress: {pct}% ({hour_num}/{total_hours} hourly bars)...")
        last_printed = pct

    hourly_date    = hourly_ts.normalize()    # the calendar date this hour belongs to
    hourly_ts_str  = hourly_ts.strftime('%Y-%m-%d %H:%M')
    hourly_date_str = hourly_date.strftime('%Y-%m-%d')

    # Find the most recent daily bar on or before this hourly timestamp's date
    available_daily = all_trading_days[all_trading_days < hourly_date]
    if available_daily.empty:
        continue
    latest_daily_date = available_daily[-1]

    for ticker in tickers:

        # -----------------------------------------------
        # RULE 1: Z-score from pre-computed cache
        # Use the Z-score of the latest available daily bar
        # -----------------------------------------------
        z_series = zscore_cache.get(ticker, pd.Series(dtype=float))
        if z_series.empty or latest_daily_date not in z_series.index:
            continue

        zscore = z_series[latest_daily_date]
        if pd.isna(zscore) or zscore > ZSCORE_THRESHOLD:
            continue

        # -----------------------------------------------
        # LAST MONTH CLOSE from pre-computed cache
        # -----------------------------------------------
        lmc_dict = last_month_close_cache.get(ticker, {})
        last_month_close = lmc_dict.get(latest_daily_date, None)
        if last_month_close is None or last_month_close == 0:
            continue

        # -----------------------------------------------
        # RULE 2: 100-hour EMA up to this hourly timestamp
        # -----------------------------------------------
        if ticker not in raw_hourly.columns.get_level_values(0):
            continue

        try:
            hourly_close = raw_hourly[ticker]['Close'].dropna()
            # Only bars up to and including this hourly timestamp
            hourly_slice = hourly_close[hourly_close.index <= hourly_ts]

            if len(hourly_slice) < EMA_PERIOD:
                continue

            ema_100    = hourly_slice.ewm(span=EMA_PERIOD, adjust=False).mean()
            latest_ema = float(ema_100.iloc[-1])
            target     = EMA_THRESHOLD * last_month_close

            if latest_ema >= target:
                continue

        except:
            continue

        # -----------------------------------------------
        # Both rules passed — record signal
        # -----------------------------------------------
        all_signals.append({
            'DateTime'        : hourly_ts_str,
            'Date'            : hourly_date_str,
            'Ticker'          : ticker,
            'Z_Score'         : round(float(zscore), 4),
            'Last_Month_Close': round(last_month_close, 2),
            'EMA_100h'        : round(latest_ema, 2),
            'Target_90pct'    : round(target, 2),
        })

# -----------------------------
# SAVE BACKTEST OUTPUT
# -----------------------------
print(f"\n{'='*60}")
if all_signals:
    df_out = pd.DataFrame(all_signals)
    df_out.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Backtest complete.")
    print(f"   Total signals    : {len(all_signals)}")
    print(f"   Unique tickers   : {df_out['Ticker'].nunique()}")
    print(f"   Date range       : {df_out['Date'].min()} to {df_out['Date'].max()}")
    print(f"   Saved to         : {OUTPUT_FILE}")

    print(f"\nTop 20 most frequent signal dates:")
    print(df_out.groupby('Date').size().rename('Signals').sort_values(ascending=False).head(20).to_string())

    print(f"\nTop 10 most frequently signalled tickers:")
    print(df_out.groupby('Ticker').size().rename('Signals').sort_values(ascending=False).head(10).to_string())
else:
    print("⚠️  No signals found across the entire backtest period.")
    print("   Check that your CSV files overlap in date range and have enough data.")