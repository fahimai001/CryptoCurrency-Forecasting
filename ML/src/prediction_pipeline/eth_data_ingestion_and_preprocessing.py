import os
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data_Preprocessing')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.Data_Preprocessing.data_ingestion import fetch_data, save_data_to_csv

SYMBOLS = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT"
}
INTERVAL = "1d"

RAW_DATA_FOLDER = "../../data/raw_data"
PROCESSED_DATA_FOLDER = "../../data/processed_data"
os.makedirs(RAW_DATA_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_DATA_FOLDER, exist_ok=True)

def load_data(filepath):
    """Loads data from a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Preprocess the data by handling missing values and other cleaning steps."""
    df = df.dropna()
    df = df.drop(columns=["ignore"], errors="ignore") 
    return df

def save_preprocessed_data(df, filename):
    """Saves the preprocessed data to a CSV file."""
    filepath = os.path.join(PROCESSED_DATA_FOLDER, filename)
    df.to_csv(filepath, index=False)
    print(f"Preprocessed data saved to {filepath}")

def process_currency_data(currency_name, symbol):
    """Fetch, preprocess, and save data for a given cryptocurrency."""
    raw_filepath = os.path.join(RAW_DATA_FOLDER, f"{currency_name}.csv")
    processed_filename = f"{currency_name}_processed.csv"
    
    if not os.path.exists(raw_filepath):
        print(f"Raw {currency_name} data not found. Fetching new data...")
        data = fetch_data(symbol, INTERVAL)
        save_data_to_csv(data, raw_filepath)
    
    print(f"Loading raw {currency_name} data...")
    raw_df = load_data(raw_filepath)
    
    print(f"Preprocessing {currency_name} data...")
    processed_df = preprocess_data(raw_df)
    
    save_preprocessed_data(processed_df, processed_filename)

def main():
    process_currency_data("bitcoin", "BTCUSDT")
    process_currency_data("ethereum", "ETHUSDT")

if __name__ == "__main__":
    main()