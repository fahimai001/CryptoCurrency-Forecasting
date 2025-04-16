import pandas as pd
from pathlib import Path

def process_dataset(input_filename: str, output_filename: str) -> None:
    """
    Process a cryptocurrency dataset by adding target column and saving to processed folder
    
    Args:
        input_filename: Name of input file in interim folder
        output_filename: Name of output file in processed folder
    """
    interim_path = Path('data/interim') / input_filename
    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(interim_path)
    df['target'] = df['Close'].shift(-1)
    
    output_path = processed_dir / output_filename
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

def main():
    """Process both Bitcoin and Ethereum datasets"""
    process_dataset(
        input_filename='bitcoin_processed.csv',
        output_filename='final_bitcoin.csv'
    )
    
    process_dataset(
        input_filename='ethereum_processed.csv',
        output_filename='final_ethereum.csv'
    )

if __name__ == '__main__':
    main()