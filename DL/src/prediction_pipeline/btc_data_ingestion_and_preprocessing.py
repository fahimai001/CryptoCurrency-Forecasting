import os
import requests
import pandas as pd
from datetime import datetime, timedelta

SYMBOLS = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT"
}
INTERVAL = "1d"  

BASE_URL = "https://api.binance.com/api/v3/klines"


RAW_DATA_FOLDER = "../../data/raw_data"
os.makedirs(RAW_DATA_FOLDER, exist_ok=True)

def fetch_data(symbol, interval, start_time, end_time, limit=1000):
    """Fetches daily data from Binance API using startTime and endTime."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "startTime": start_time,
        "endTime": end_time
    }
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status() 
        data = response.json()
        
        if not data:
            print(f"Warning: No data returned for {symbol}. Check API limits or dates.")
            return None
        
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def preprocess_data(data):
    """Converts raw API data into a structured DataFrame."""
    columns = [
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]
    
    if data is None or len(data) == 0:
        print("Error: Received empty or invalid data for preprocessing.")
        return None
    
    df = pd.DataFrame(data, columns=columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    print(f"Preprocessing completed. Data shape: {df.shape}")
    return df

def save_data_to_csv(df, filename):
    """Saves the DataFrame to a CSV file."""
    if df is None or df.empty:
        print(f"Error: No valid data to save for {filename}. Skipping.")
        return
    
    df.to_csv(filename, index=False)
    print(f"Data successfully saved to {filename}")

def main():
    start_date = datetime(2024, 2, 1) 
    end_date = datetime(2025, 2, 4) 
    start_time = int(start_date.timestamp() * 1000)
    end_time = int(end_date.timestamp() * 1000)
    
    for name, symbol in SYMBOLS.items():
        print(f"Fetching data for {name} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        raw_data = fetch_data(symbol, INTERVAL, start_time, end_time)

        if raw_data is None:
            print(f"Skipping {name} due to API error or empty data.")
            continue

        processed_df = preprocess_data(raw_data)

        if processed_df is None:
            print(f"Skipping {name} due to processing error.")
            continue

        save_data_to_csv(processed_df, os.path.join(RAW_DATA_FOLDER, f"{name}.csv"))

if __name__ == "__main__":
    main()
