import pandas as pd
import numpy as np
import os

# -----------------------------
# CONFIG
# -----------------------------
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
    df["EMA100"] = df["Close"].ewm(span=100, adjust=False).mean()
    return df

#def get_monthly_close_above_ema(df):
    # Get last Close and EMA100 of each month
 #   monthly_close = df["Close"].resample("M").last()
 #   monthly_benchmark = (monthly_close * 0.9)
  #  daily_benchmark = monthly_benchmark.reindex(df.index, method="ffill")
  #  flag = (daily_benchmark > df["EMA100"]).map({True: "Yes", False: "No"})
  #  return flag


def get_monthly_close_above_ema(df):
    # Get last Close and EMA100 of each month
    monthly_close = df["Close"].resample("M").last()
    monthly_ema   = df["EMA100"].resample("M").last()
    # Compare: is monthly close > monthly EMA100?
    monthly_benchmark = (monthly_close * 0.9)
    monthly_flag  = (monthly_benchmark > monthly_ema).map({True: "Yes", False: "No"})
    # Reindex back to daily index (forward-fill so every day in a month carries that month's verdict)
    daily_flag = monthly_flag.reindex(df.index, method="ffill")
    return daily_flag 


# -----------------------------
# CALCULATE ZSCORE + EMA FILTER
# -----------------------------
data = []

for ticker in tickers:
    try:
        df = get_ticker_df(ticker)
        if df.empty:
            continue

        # Compute daily indicators
        df = compute_indicators(df)

        # Add monthly close vs EMA200 column
        df["Above_EMA100"] = get_monthly_close_above_ema(df)

        # Filter: only keep rows where ZScore <= -1.4
        df_filtered = df[df["ZScore"] <= -1.4]

        if df_filtered.empty:
            continue

        temp_df = pd.DataFrame({
            "Date":        df_filtered.index,
            "Ticker":      ticker,
            "Close":       df_filtered["Close"].round(2).values,
            "EMA100":      df_filtered["EMA100"].round(2).values,
            "ZScore":      df_filtered["ZScore"].round(2).values,
            "Above_EMA100": df_filtered["Above_EMA100"].values,
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
