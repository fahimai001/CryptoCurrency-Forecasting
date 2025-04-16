"""
Script to collect historical data for Bitcoin and Ethereum from Binance API
"""
import os
import pandas as pd
import requests
from datetime import datetime
import time

def get_binance_klines(symbol, interval='1d', start_time=None, end_time=None):
    """
    Fetch klines/candlestick data from Binance API
    """
    base_url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': 1000
    }

    if start_time:
        params['startTime'] = start_time
    if end_time:
        params['endTime'] = end_time

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data: {response.status_code}")
        print(response.text)
        return []

def fetch_all_historical_data(symbol, interval='1d', start_date="2017-01-01"):
    """
    Fetch all historical data from start_date to now
    """
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.now().timestamp() * 1000)

    all_klines = []
    current_start = start_ts

    while current_start < end_ts:
        klines = get_binance_klines(symbol, interval, current_start, end_ts)
        if not klines:
            break

        all_klines.extend(klines)
        current_start = klines[-1][0] + 1
        time.sleep(0.5)

        print(f"Fetched {len(klines)} klines for {symbol}. Total: {len(all_klines)}")

    columns = [
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
    ]

    df = pd.DataFrame(all_klines, columns=columns)
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col])

    df.set_index('Open time', inplace=True)
    return df

def save_data(df, filename, directory="data/external"):
    """
    Save DataFrame to CSV file inside the given directory
    """
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    df.to_csv(filepath)
    print(f"Data saved to {filepath}")
    return filepath

def main():
    """
    Main function to collect and save cryptocurrency data
    """
    start_date = "2017-01-01"
    data_dir = os.path.join("data", "external")

    print("Fetching Bitcoin data...")
    btc_df = fetch_all_historical_data("BTCUSDT", start_date=start_date)
    save_data(btc_df, "bitcoin.csv", data_dir)

    print("Fetching Ethereum data...")
    eth_df = fetch_all_historical_data("ETHUSDT", start_date=start_date)
    save_data(eth_df, "ethereum.csv", data_dir)

    print("Data collection completed.")

if __name__ == "__main__":
    main()
