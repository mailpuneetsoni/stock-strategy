import io
import pandas as pd
import requests
import yfinance as yf


# --- Step 1: Load Nifty tickers from CSV ---
csv_path = r"D:\Stock_Strategy\EQUITY__nifty.csv"
try:
    df_nifty = pd.read_csv(csv_path)
except Exception as e:
    raise SystemExit(f"Failed to read CSV at {csv_path}: {e}")

# detect common ticker column names, fallback to first column
_possible_cols = ['Symbol', 'SYMBOL', 'symbol', 'Ticker', 'TICKER', 'ticker']
_col = next((c for c in _possible_cols if c in df_nifty.columns), df_nifty.columns[0])

# extract, clean, dedupe tickers
_raw_tickers = df_nifty[_col].dropna().astype(str).str.strip().unique().tolist()

# ensure Yahoo Finance NSE format (append .NS if missing)
def _ensure_ns(t):
    return t if t.upper().endswith('.NS') else f"{t}.NS"

ticker_universe = [_ensure_ns(t) for t in _raw_tickers if t != '']

# --- Step 2: Set your Market Cap threshold ---
# IMPORTANT NOTE: yfinance returns the market cap in the asset's local currency.
# For NSE cash stocks, this value is explicitly in Indian Rupees (INR).
# Example: 10_000_000_000 INR represents a market cap greater than ₹1,000 Crores (10 Billion INR).
MARKET_CAP_THRESHOLD = 10_000_000_000 

filtered_stocks = []

print("Scanning tickers via Yahoo Finance... This will take a few minutes for 500 stocks.")

# --- Step 3: Loop through the universe and extract market statistics ---
for index, ticker in enumerate(ticker_universe, start=1):
    try:
        # Visual progress update indicator every 50 tickers
        if index % 50 == 0 or index == len(ticker_universe):
            print(f"Processed {index}/{len(ticker_universe)} tickers...")

        stock = yf.Ticker(ticker)
        info = stock.info
        market_cap = info.get('marketCap', 0)
        
        # Filter condition
        if market_cap and market_cap > MARKET_CAP_THRESHOLD:
            filtered_stocks.append({
                "Ticker": ticker
            })
    except Exception as e:
        # Skip individual tickers that throw timeout or API errors
        continue

# --- Step 4: Format and present the final list ---
if filtered_stocks:
    df = pd.DataFrame(filtered_stocks)
    print("\n--- Matching Stocks Found ---")
    print(df.to_string(index=False))
    
    # Save output data to a local CSV file
    df.to_csv(r"D:\Stock_Strategy\filtered_nifty_market_cap_stocks.csv", index=False)
    print("\nResults successfully saved to 'filtered_nifty_market_cap_stocks.csv'")
else:
    print("\nNo stocks matched the specified criteria.")