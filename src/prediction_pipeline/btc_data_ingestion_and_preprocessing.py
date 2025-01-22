import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data_Preprocessing')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
from src.Data_Preprocessing.data_ingestion import fetch_data, save_data_to_csv

# Define the paths
RAW_DATA_FOLDER = "../../data/raw_data"
PROCESSED_DATA_FOLDER = "../../data/processed_data"
os.makedirs(PROCESSED_DATA_FOLDER, exist_ok=True)

def load_data(filepath):
    """Loads data from a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Preprocess the data by handling missing values and other cleaning steps."""
    # Drop rows with missing values
    df = df.dropna()
    
    # Additional preprocessing steps can be added here
    df = df.drop(columns=["ignore"], errors="ignore")  # Drop unnecessary columns
    df["timestamp"] = pd.to_datetime(df["timestamp"])  # Ensure timestamp is datetime
    
    return df

def save_preprocessed_data(df, filename):
    """Saves the preprocessed data to a CSV file."""
    filepath = os.path.join(PROCESSED_DATA_FOLDER, filename)
    df.to_csv(filepath, index=False)
    print(f"Preprocessed data saved to {filepath}")

def main():
    # Fetch Bitcoin data using the data_ingestion script
    bitcoin_filepath = os.path.join(RAW_DATA_FOLDER, "bitcoin.csv")
    
    # If the raw data doesn't exist, fetch and save it
    if not os.path.exists(bitcoin_filepath):
        print("Raw Bitcoin data not found. Fetching new data...")
        data = fetch_data("BTCUSDT", "1h")
        save_data_to_csv(data, bitcoin_filepath)
    
    # Load the raw Bitcoin data
    print("Loading raw Bitcoin data...")
    raw_df = load_data(bitcoin_filepath)
    
    # Preprocess the data
    print("Preprocessing Bitcoin data...")
    processed_df = preprocess_data(raw_df)
    
    # Save the preprocessed data
    save_preprocessed_data(processed_df, "bitcoin_processed.csv")

if __name__ == "__main__":
    main()