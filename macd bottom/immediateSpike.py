import pandas as pd
import numpy as np
import os

# -----------------------------
# CONFIG
# -----------------------------
MARKET_CAP_THRESHOLD = 1e10
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
CSV_FILE     = os.path.join(desktop_path, 'stock_data.csv')
OUTPUT_FILE  = os.path.join(desktop_path, 'monthly_close.csv')

# -----------------------------
# LOAD CSV
# -----------------------------
raw = pd.read_csv(CSV_FILE, header=[0, 1], index_col=0, parse_dates=True)
tickers = raw.columns.get_level_values(0).unique().tolist()

def get_ticker_df(ticker):
    df = raw[ticker]
    df = df.dropna(how='all')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def compute_indicators(df):
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["STD200"] = df["Close"].rolling(200).std()
    df["ZScore"] = (df["Close"] - df["SMA200"]) / df["STD200"]
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["SMA50"]  = df["Close"].rolling(50).mean()
    return df

# -----------------------------
# CALCULATE ZSCORE
# -----------------------------
data = []

for ticker in tickers:
    try:
        df = get_ticker_df(ticker)
        if df.empty:
            continue

        # Compute daily indicators (ZScore is daily)
        df = compute_indicators(df)

        # Resample both Close and ZScore to month-end (last value of each month)
       # monthly_close  = df["Close"].resample("M").last()
        zscore = df["ZScore"] # 

        temp_df = pd.DataFrame({
            "Date":          zscore.index,
            "Ticker":        ticker,
            "ZScore":        zscore.round(2).values,
        })

        data.append(temp_df)

    except Exception as e:
        print(f"Error processing {ticker}: {e}")

# -----------------------------
# SAVE TO CSV
# -----------------------------
if data:
    final_df = pd.concat(data, ignore_index=True)
    final_df = final_df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")
    print(f"Shape: {final_df.shape}")
    print(final_df.tail(10))
else:
    print("No data to save.")
