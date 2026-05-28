import pandas as pd
import numpy as np
import os

# -----------------------------
# CONFIG
# -----------------------------
desktop_path  = os.path.join(os.path.expanduser('~'), 'Desktop')
DAILY_FILE    = os.path.join(desktop_path, 'stock_data_daily.csv')   
HOURLY_FILE   = os.path.join(desktop_path, 'stock_data.csv')          
OUTPUT_FILE   = os.path.join(desktop_path, 'backtest_signals.csv')    
LOG_FILE      = os.path.join(desktop_path, 'backtest_logs.csv')        

LOOKBACK_DAYS = 365
ZSCORE_WINDOW    = 200
ZSCORE_THRESHOLD = -1.4
EMA_PERIOD       = 100
EMA_THRESHOLD    = 0.90

# -----------------------------
# DATA LOADING
# -----------------------------
print("Loading daily data...")
raw_daily = pd.read_csv(DAILY_FILE, header=[0, 1], index_col=0, parse_dates=True)
raw_daily.index = pd.to_datetime(raw_daily.index).tz_localize(None)   
tickers = raw_daily.columns.get_level_values(0).unique().tolist()

print("Loading hourly data...")
raw_hourly = pd.read_csv(HOURLY_FILE, header=[0, 1], index_col=0, parse_dates=True)
raw_hourly.index = pd.to_datetime(raw_hourly.index).tz_localize(None)  

# Initialize log array early to catch validation rejections
all_rows_log = []  # Holds complete matrix data

# -----------------------------
# DATA VALIDATION FUNCTION
# -----------------------------
def validate_ticker_data(ticker, raw_daily, raw_hourly, z_window=200, ema_period=100):
    """
    Validates a ticker's data for consistency, structural integrity, and inaccuracies.
    Returns: (bool, reason_string)
    """
    # 1. STRUCTURAL CHECK: Does the ticker exist in both dataframes?
    if ticker not in raw_daily.columns.get_level_values(0):
        return False, "Missing entirely from Daily CSV columns"
    if ticker not in raw_hourly.columns.get_level_values(0):
        return False, "Missing entirely from Hourly CSV columns"

    try:
        # Extract closing prices and drop missing entries
        d_close = raw_daily[ticker]['Close'].dropna()
        h_close = raw_hourly[ticker]['Close'].dropna()
        
        # 2. SUFFICIENCY CHECK: Is there enough data to compute technical indicators?
        if len(d_close) < z_window:
            return False, f"Insufficient daily data (Has {len(d_close)} rows, needs {z_window} for Z-Score)"
        if len(h_close) < ema_period:
            return False, f"Insufficient hourly data (Has {len(h_close)} rows, needs {ema_period} for EMA)"

        # 3. ACCURACY CHECK: Are there corrupted/impossible price points?
        if (d_close <= 0).any() or (h_close <= 0).any():
            return False, "Data corruption: Contains zero or negative closing prices"

        # 4. CONSISTENCY CHECK: Do extreme, unrealistic single-day data spikes exist?
        daily_pct_changes = d_close.pct_change().abs()
        if (daily_pct_changes > 4.0).any():  # Flags greater than a 400% price movement in one day
            return False, "Data anomaly: Contains an unrealistic price spike/drop (>400% in one day)"

        # 5. ALIGNMENT CHECK: Do the daily and hourly data timelines actually overlap?
        d_min, d_max = d_close.index.min(), d_close.index.max()
        h_min, h_max = h_close.index.min(), h_close.index.max()

        if h_max < d_min or d_max < h_min:
            return False, f"Timeline Disconnect: Daily range ({d_min.strftime('%Y-%m-%d')} to {d_max.strftime('%Y-%m-%d')}) does not overlap Hourly range"

    except KeyError:
        return False, "Missing required 'Close' sub-column identifier under this ticker"
    except Exception as e:
        return False, f"Unexpected data parsing error: {str(e)}"

    return True, "Passed Integrity Check"

# -----------------------------
# RUN DATA INTEGRITY FILTERING
# -----------------------------
print("\nRunning data integrity and consistency tests...")
valid_tickers = []

for ticker in tickers:
    passed, reason = validate_ticker_data(ticker, raw_daily, raw_hourly, ZSCORE_WINDOW, EMA_PERIOD)
    
    if passed:
        valid_tickers.append(ticker)
    else:
        print(f"  ⚠️ Ticker '{ticker}' REJECTED | Reason: {reason}")
        # Append to matrix log so it appears in your final CSV spreadsheet
        all_rows_log.append({
            'DateTime': 'INITIALIZATION_PHASE',
            'Ticker': ticker,
            'Z_Score': None,
            'Rule1_Passed': False,
            'Hourly_Close': None,
            'Last_Month_Close': None,
            'Target_90pct': None,
            'EMA_100h': None,
            'Rule2_Passed': False,
            'Status': f"REJECTED: {reason}"
        })

print(f"-> Integrity checks complete. Proceeding with {len(valid_tickers)} out of {len(tickers)} tickers.\n")
tickers = valid_tickers  # Update active ticker pool to only include verified clean data

# -----------------------------
# UPFRONT PRE-COMPUTATION
# -----------------------------
print("[Pre-computing] Daily Z-Scores...")
zscore_cache = {}   
for ticker in tickers:
    try:
        close = raw_daily[ticker]['Close'].dropna()
        if len(close) >= ZSCORE_WINDOW:
            sma = close.rolling(ZSCORE_WINDOW).mean()
            std = close.rolling(ZSCORE_WINDOW).std().replace(0, np.nan) 
            zscore_cache[ticker] = (close - sma) / std
    except:
        pass

print("[Pre-computing] Daily Last Month Closes...")
last_month_close_cache = {}   
for ticker in tickers:
    try:
        close = raw_daily[ticker]['Close'].dropna()
        if not close.empty:
            monthly_last = close.resample('ME').last()   
            result = {}
            for date in close.index:
                prev_month_end = (date.replace(day=1) - pd.DateOffset(days=1))
                prev_month_key = prev_month_end.to_period('M').to_timestamp('M')
                available = monthly_last[monthly_last.index <= prev_month_key]
                result[date] = float(available.iloc[-1]) if not available.empty else None
            last_month_close_cache[ticker] = result
    except:
        pass

print("[Pre-computing] Hourly Closes & 100h EMAs...")
hourly_close_cache = {}
hourly_ema_cache = {}
for ticker in tickers:
    try:
        h_close = raw_hourly[ticker]['Close'].dropna()
        hourly_close_cache[ticker] = h_close
        hourly_ema_cache[ticker] = h_close.ewm(span=EMA_PERIOD, adjust=False).mean()
    except:
        pass

# -----------------------------
# TIMELINE SETUP
# -----------------------------
all_trading_days = raw_daily.index.sort_values()
min_history_date = all_trading_days[ZSCORE_WINDOW] if len(all_trading_days) > ZSCORE_WINDOW else all_trading_days[0]
cutoff_date      = all_trading_days[-1] - pd.DateOffset(days=LOOKBACK_DAYS)
start_date       = max(cutoff_date, min_history_date)

hourly_timestamps = raw_hourly.index.sort_values()
backtest_hourly   = hourly_timestamps[hourly_timestamps >= start_date]

# -----------------------------
# BACKTEST & MATRIX LOGGING LOOP
# -----------------------------
print(f"\nStarting backtest loop across {len(backtest_hourly)} hourly bars...")
all_signals = []

for hourly_ts in backtest_hourly:
    hourly_date = hourly_ts.normalize()    
    hourly_ts_str = hourly_ts.strftime('%Y-%m-%d %H:%M')

    # Avoid look-ahead bias by pulling the prior completed daily session
    available_daily = all_trading_days[all_trading_days < hourly_date]
    if available_daily.empty:
        continue
    latest_daily_date = available_daily[-1]

    for ticker in tickers:
        # Pull parameters safely (defaulting to NaN if data is missing)
        z_series = zscore_cache.get(ticker, pd.Series(dtype=float))
        zscore = z_series.get(latest_daily_date, np.nan)
        
        lmc_dict = last_month_close_cache.get(ticker, {})
        last_month_close = lmc_dict.get(latest_daily_date, np.nan)
        
        h_close_series = hourly_close_cache.get(ticker, pd.Series(dtype=float))
        hourly_close_val = h_close_series.get(hourly_ts, np.nan)
        
        h_ema_series = hourly_ema_cache.get(ticker, pd.Series(dtype=float))
        hourly_ema_val = h_ema_series.get(hourly_ts, np.nan)
        
        # Calculate target threshold (LMC * 0.90)
        target_val = EMA_THRESHOLD * last_month_close if pd.notna(last_month_close) else np.nan

        # Evaluate Rule Conditions
        r1_passed = False
        if pd.notna(zscore) and zscore <= ZSCORE_THRESHOLD:
            r1_passed = True

        r2_passed = False
        if pd.notna(hourly_ema_val) and pd.notna(target_val) and (hourly_ema_val < target_val):
            r2_passed = True

        # Determine Row Classification Status
        if pd.isna(zscore) or pd.isna(last_month_close) or pd.isna(hourly_ema_val):
            status = "MISSING_DATA"
        elif r1_passed and r2_passed:
            status = "SIGNAL_MATCH"
        elif r1_passed and not r2_passed:
            status = "FAIL_RULE_2"
        else:
            status = "FAIL_RULE_1"

        # Build complete parameter log entry
        log_entry = {
            'DateTime': hourly_ts_str,
            'Ticker': ticker,
            'Z_Score': round(zscore, 4) if pd.notna(zscore) else None,
            'Rule1_Passed': r1_passed,
            'Hourly_Close': round(hourly_close_val, 2) if pd.notna(hourly_close_val) else None,
            'Last_Month_Close': round(last_month_close, 2) if pd.notna(last_month_close) else None,
            'Target_90pct': round(target_val, 2) if pd.notna(target_val) else None,
            'EMA_100h': round(hourly_ema_val, 2) if pd.notna(hourly_ema_val) else None,
            'Rule2_Passed': r2_passed,
            'Status': status
        }
        all_rows_log.append(log_entry)

        # Record winning signals
        if status == "SIGNAL_MATCH":
            all_signals.append({
                'DateTime': hourly_ts_str,
                'Date': hourly_date.strftime('%Y-%m-%d'),
                'Ticker': ticker,
                'Z_Score': log_entry['Z_Score'],
                'Last_Month_Close': log_entry['Last_Month_Close'],
                'EMA_100h': log_entry['EMA_100h'],
                'Target_90pct': log_entry['Target_90pct'],
            })

# -----------------------------
# SAVE CSV EXPORTS
# -----------------------------
print(f"\n{'='*60}")
print("Writing files to Desktop...")

df_logs = pd.DataFrame(all_rows_log)
df_logs.to_csv(LOG_FILE, index=False)
print(f"📊 Audit Log Matrix exported to: {LOG_FILE} ({len(df_logs):,} items matrixed)")

if all_signals:
    df_out = pd.DataFrame(all_signals)
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Strategy Signals exported to: {OUTPUT_FILE} ({len(all_signals)} records found)")
else:
    print("⚠️ 0 strategic matches tracked in the signals file.") 