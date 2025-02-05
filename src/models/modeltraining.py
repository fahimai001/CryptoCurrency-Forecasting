import sys
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import pickle

# Add the paths to your 'src' directory correctly
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the absolute path of the current script
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))  # Go up two levels to the project root

# Add 'src' directory to the path
sys.path.insert(0, os.path.join(project_root, 'src'))

# Now you can import your modules
from prediction_pipeline.btc_data_ingestion_and_preprocessing import preprocess_data as preprocess_btc
from prediction_pipeline.eth_data_ingestion_and_preprocessing import preprocess_data as preprocess_eth

# Define the paths
RAW_DATA_FOLDER = "../../data/raw_data"
PROCESSED_DATA_FOLDER = "../../data/processed_data"
ARTIFACTS_FOLDER = "../../artifacts"

# Create folders if they don't exist
os.makedirs(ARTIFACTS_FOLDER, exist_ok=True)

def load_data(filepath):
    """Loads data from a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    return pd.read_csv(filepath)

def preprocess_and_split_data(currency_name, preprocess_func):
    """Preprocess data and split into train and test sets."""
    print(f"Processing {currency_name} data...")
    raw_data = load_data(os.path.join(PROCESSED_DATA_FOLDER, f"{currency_name}_processed.csv"))
    
    # Preprocess data
    processed_data = preprocess_func(raw_data)
    
    # Select features and target
    X = processed_data.drop(columns=["close"], errors="ignore")
    y = processed_data["close"]
    
    # Split into train and test
    split_index = int(len(processed_data) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Return unmodified data (no scaling applied)
    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_type, model_name):
    """Train a model, evaluate and save it to Artifacts."""
    # Train model
    if model_type == "linear_regression":
        model = LinearRegression()
    elif model_type == "xgboost":
        model = xgb.XGBRegressor(objective="reg:squarederror")
    else:
        raise ValueError("Invalid model type. Choose 'linear_regression' or 'xgboost'.")
    
    # Fit the model
    model.fit(X_train, y_train.ravel())
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"{model_name} - MAE: {mae}, MSE: {mse}, RMSE: {rmse}")
    
    # Save the model to Artifacts
    model_filepath = os.path.join(ARTIFACTS_FOLDER, f"{model_name}.pkl")
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"{model_name} saved to {model_filepath}")

def main():
    try:
        # Process and prepare data for Bitcoin
        X_train_btc, X_test_btc, y_train_btc, y_test_btc = preprocess_and_split_data("bitcoin", preprocess_btc)
        
        # Process and prepare data for Ethereum
        X_train_eth, X_test_eth, y_train_eth, y_test_eth = preprocess_and_split_data("ethereum", preprocess_eth)
        
        # Train and evaluate models for Bitcoin data
        print("Training and evaluating Linear Regression on Bitcoin data...")
        train_and_evaluate_model(X_train_btc, X_test_btc, y_train_btc, y_test_btc, "linear_regression", "linear_regression_btc")
        print("Training and evaluating XGBoost on Bitcoin data...")
        train_and_evaluate_model(X_train_btc, X_test_btc, y_train_btc, y_test_btc, "xgboost", "xgboost_btc")
        
        # Train and evaluate models for Ethereum data
        print("Training and evaluating Linear Regression on Ethereum data...")
        train_and_evaluate_model(X_train_eth, X_test_eth, y_train_eth, y_test_eth, "linear_regression", "linear_regression_eth")
        print("Training and evaluating XGBoost on Ethereum data...")
        train_and_evaluate_model(X_train_eth, X_test_eth, y_train_eth, y_test_eth, "xgboost", "xgboost_eth")

    except FileNotFoundError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
