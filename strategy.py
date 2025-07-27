import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

  
def fetch_and_calculate_macd(symbol, period="10y"):
    try:
        print(f"Fetching data for {symbol}...")
        
        # Fetch stock data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval="1wk")
        
        if data.empty:
            print(f"No data found for symbol: {symbol}")
            return None
        
        # Reset index to make Date a column
        data.reset_index(inplace=True)
        
        # Create clean dataframe with essential columns
        df = pd.DataFrame({
            'Date': data['Date'],
            'Close': data['Close'].round(2)
        })
        
        # Calculate MACD with standard settings
        # Fast EMA (12 periods)
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        
        # Slow EMA (26 periods)  
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD Line = EMA(12) - EMA(26)
        df['MACD_Line'] = (ema_12 - ema_26).round(4)
        
        # Signal Line = EMA(9) of MACD Line
        df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean().round(4)
        
        # MACD Histogram = MACD Line - Signal Line
        df['MACD_Histogram'] = (df['MACD_Line'] - df['MACD_Signal']).round(4)
        
        # Add additional useful columns
        df['MACD_Signal_Direction'] = df['MACD_Line'].diff().apply(lambda x: 'Up' if x > 0 else 'Down' if x < 0 else 'Flat')
        
        # Identify crossovers
        df['MACD_Crossover'] = ''
        for i in range(1, len(df)):
            prev_hist = df['MACD_Histogram'].iloc[i-1]
            curr_hist = df['MACD_Histogram'].iloc[i]
            
            if prev_hist <= 0 and curr_hist > 0:
                df.loc[i, 'MACD_Crossover'] = 'Bullish'
            elif prev_hist >= 0 and curr_hist < 0:
                df.loc[i, 'MACD_Crossover'] = 'Bearish'
        
        # Calculate 10-week rolling mean and standard deviation for MACD Line
        df['MACD_Rolling_Mean_10'] = df['MACD_Line'].rolling(window=10, min_periods=10).mean().round(4)
        df['MACD_Rolling_Std_10'] = df['MACD_Line'].rolling(window=10, min_periods=10).std().round(4)
        
        # Identify overbought and oversold conditions
        df['Overbought'] = (df['MACD_Line'] > (df['MACD_Rolling_Mean_10'] + 2 * df['MACD_Rolling_Std_10'])).astype(bool)
        df['Oversold'] = (df['MACD_Line'] < (df['MACD_Rolling_Mean_10'] - 2 * df['MACD_Rolling_Std_10'])).astype(bool)
        
        # Add symbol column
        df['Symbol'] = symbol
        
        print(f"Successfully calculated MACD for {len(df)} records")
        return df
        
    except Exception as e:
        print(f"Error processing data for {symbol}: {str(e)}")
        return None

def save_combined_macd_data(data, filename_prefix="Nifty50_Weekly_MACD_Data"):
    """
    Save combined MACD data to a single CSV file
    """
    if data is None or data.empty:
        return None
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{filename_prefix}_{timestamp}.csv"
    
    try:
        data.to_csv(filename, index=False)
        
        print(f"\n{'='*60}")
        print(f"FILE SAVED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Filename: {filename}")
        print(f"Records: {len(data)}")
        print(f"Columns: {', '.join(data.columns)}")
        
        # File size
        import os
        file_size = os.path.getsize(filename) / 1024
        print(f"File Size: {file_size:.2f} KB")
        
        absolute_path = os.path.abspath(filename)
        print(f"Using os.path.abspath: {absolute_path}")
        
        return filename
        
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        return None   

# Removed duplicate save_macd_data and display_macd_summary for brevity, as they are not used in combined mode

if __name__ == "__main__":
    print("=== Fetching Data for All Nifty 50 Shares (Previous 10 Years) ===")
    
    # List of Nifty 50 symbols (with .NS suffix for yfinance)
    nifty50_symbols = [
        "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
        "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
        "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "DIVISLAB.NS",
        "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS",
        "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS",
        "ITC.NS", "JIOFIN.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
        "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
        "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SHRIRAMFIN.NS", "SBIN.NS",
        "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
        "TECHM.NS", "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS"
    ]
    
    all_macd_data = []
    
    for symbol in nifty50_symbols:
        print(f"Fetching 10 years of data for {symbol}...")
        macd_data = fetch_and_calculate_macd(symbol, period="10y")
            
        if macd_data is not None:
            all_macd_data.append(macd_data)
            print(f"✅ MACD data successfully generated for {symbol}!")
        
        else:
            print(f"❌ Failed to generate MACD data for {symbol}")
            print("Please check your internet connection and try again.")
    
    if all_macd_data:
        combined_data = pd.concat(all_macd_data, ignore_index=True)
        saved_file = save_combined_macd_data(combined_data)
        
        if saved_file:
            print(f"\n✅ Combined MACD data for all Nifty 50 stocks successfully saved!")
            print(f"📊 Use this file for technical analysis and trading strategies")
            print(f"📈 MACD settings: Fast(12), Slow(26), Signal(9)")
    else:
        print("❌ No data was generated for any symbols.")