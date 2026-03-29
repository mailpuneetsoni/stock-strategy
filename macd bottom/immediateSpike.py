import pandas as pd
import numpy as np
import os

# -----------------------------
# CONFIG
# -----------------------------
MARKET_CAP_THRESHOLD = 1e10  # ₹1000 Cr

desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
CSV_FILE = os.path.join(desktop_path, 'stock_data.csv')

# -----------------------------
# LOAD CSV
# -----------------------------
raw = pd.read_csv(CSV_FILE, header=[0, 1], index_col=0, parse_dates=True)
# raw has MultiIndex columns: (field, ticker) e.g. ('Close', 'RELIANCE.NS')

tickers = raw.columns.get_level_values(1).unique().tolist()

# -----------------------------
# FUNCTIONS
# -----------------------------
def get_ticker_df(ticker):
    """Extract OHLCV DataFrame for a single ticker from the multi-index CSV."""
    df = raw.xs(ticker, axis=1, level=1)  # slice columns for this ticker
    df = df.dropna(how='all')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def compute_indicators(df):
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["STD200"] = df["Close"].rolling(200).std()
    df["ZScore"] = (df["Close"] - df["SMA200"]) / df["STD200"]
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["SMA50"] = df["Close"].rolling(50).mean() # seems redundant but used in condition, so keeping it here
    return df


def monthly_close_above_ema(df):
    monthly = df["Close"].resample("ME").last()
    ema200 = monthly.ewm(span=200, adjust=False).mean()
    return monthly.iloc[-1] > ema200.iloc[-1]


def sma50_uptrend(df):
    sma = df["SMA50"]
    return sma.iloc[-1] > sma.iloc[-20]


def scan_stock(ticker):
    try:
        df = get_ticker_df(ticker)

        if df.empty or len(df) < 200:
            return None

        df = compute_indicators(df)
        latest = df.iloc[-1]

        # Market cap: still fetched live since CSV doesn't store it
        import yfinance as yf
        info = yf.Ticker(ticker).info
        market_cap = info.get("marketCap", 0)

        # -----------------------------
        # CONDITIONS
        # -----------------------------
        cond_zscore     = latest["ZScore"] < -1.4
        cond_monthly    = monthly_close_above_ema(df)
        cond_marketcap  = market_cap > MARKET_CAP_THRESHOLD
        cond_sma_trend  = sma50_uptrend(df)

        if all([cond_zscore, cond_monthly, cond_marketcap, cond_sma_trend]):
            return {
                "Ticker": ticker,
                "ZScore": round(latest["ZScore"], 2),
                "Close": round(latest["Close"], 2),
                "MarketCap": market_cap
            }

    except Exception as e:
        print(f"Error with {ticker}: {e}")

    return None


# -----------------------------
# RUN SCREENER
# -----------------------------
results = []

for ticker in tickers:
    print(f"Scanning {ticker}...")
    res = scan_stock(ticker)
    if res:
        results.append(res)

# Output
df_results = pd.DataFrame(results)
print("\n--- Screener Results ---")
print(df_results if not df_results.empty else "No stocks matched the criteria.")