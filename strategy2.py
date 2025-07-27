import yfinance as yf
import pandas as pd
from datetime import datetime

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
        
        # Add MACD Signal Direction
        df['MACD_Signal_Direction'] = df['MACD_Line'].diff().apply(lambda x: 'Up' if x > 0 else 'Down' if x < 0 else 'Flat')
        
        # Add symbol column
        df['Symbol'] = symbol
        
        print(f"Successfully calculated MACD for {len(df)} records")
        return df
        
    except Exception as e:
        print(f"Error processing data for {symbol}: {str(e)}")
        return None

def save_strategy_results(strategy_df, filename_prefix="Strategy_Results_MACD_Slope_Increasing"):
    """
    Save the strategy results to a CSV file with specified columns
    """
    if strategy_df.empty:
        print("No data to save.")
        return None
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{filename_prefix}_{timestamp}.csv"
    
    try:
        strategy_df.to_csv(filename, index=False)
        
        print(f"\n{'='*60}")
        print(f"STRATEGY RESULTS FILE SAVED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Filename: {filename}")
        print(f"Records: {len(strategy_df)}")
        print(f"Columns: {', '.join(strategy_df.columns)}")
        
        # File size
        import os
        file_size = os.path.getsize(filename) / 1024
        print(f"File Size: {file_size:.2f} KB")
        
        absolute_path = os.path.abspath(filename)
        print(f"Using os.path.abspath: {absolute_path}")
        
        return filename
        
    except Exception as e:
        print(f"Error saving strategy results file: {str(e)}")
        return None

def add_meets_strategy(combined_data, weeks=3):
    """
    Add 'Meets_Strategy' column to the DataFrame, flagging rows where the slope condition holds historically.
    Set to False by default, True where met.
    """
    combined_data['Meets_Strategy'] = False
    grouped = combined_data.groupby('Symbol')
    
    for symbol, group in grouped:
        group = group.sort_values('Date').reset_index(drop=True)
        for i in range(weeks, len(group)):
            recent_macd = group['MACD_Line'].iloc[i-weeks:i+1].values
            slopes = [recent_macd[j+1] - recent_macd[j] for j in range(weeks)]
            if all(slopes[j] < slopes[j+1] for j in range(weeks - 1)):
                # Set True for the ending row of the window
                global_idx = group.index[i]  # Local reset index, so use .index
                combined_data.at[global_idx, 'Meets_Strategy'] = True
    return combined_data

if __name__ == "__main__":
    print("=== Fetching Data for All Nifty 50 Shares (Previous 10 Years) ===")
    
    # Updated list of Nifty 50 symbols as of July 2025
    nifty50_symbols = [
        "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
        "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
        "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "ZOMATO.NS",
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
        
        # Add Meets_Strategy column historically
        combined_data = add_meets_strategy(combined_data, weeks=3)
        
        # Add shifted columns for MACD_Signal previous weeks
        combined_data['MACD_Signal_-1_week'] = combined_data.groupby('Symbol')['MACD_Signal'].shift(1)
        combined_data['MACD_Signal_-2_week'] = combined_data.groupby('Symbol')['MACD_Signal'].shift(2)
        
        # Get the full DataFrame with all rows and Meets_Strategy True/False
        strategy_df = combined_data[['Date', 'Symbol', 'Close', 'MACD_Line', 'MACD_Signal', 'MACD_Signal_Direction', 'Meets_Strategy', 'MACD_Signal_-1_week', 'MACD_Signal_-2_week']]
        
        # Save the strategy results
        saved_strategy_file = save_strategy_results(strategy_df)
        
        if saved_strategy_file:
            print(f"\n✅ Strategy results saved to CSV with all details!")
        
        # Optional: Print number of historical signals
        num_qualifying = combined_data['Meets_Strategy'].sum()
        print(f"\nTotal historical qualifying instances: {num_qualifying}")
    else:
        print("❌ No data was generated for any symbols.")