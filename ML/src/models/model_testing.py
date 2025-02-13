import sys
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import pickle

RAW_DATA_FOLDER = "../../data/raw_data"
PROCESSED_DATA_FOLDER = "../../data/processed_data"
ARTIFACTS_FOLDER = "../../artifacts"

os.makedirs(ARTIFACTS_FOLDER, exist_ok=True)

def load_data(filepath):
    """Loads data from a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    return pd.read_csv(filepath)

def preprocess_data(data):
    """Preprocessing example (replace with your actual preprocessing)."""

    data = data.dropna()  
    return data

def preprocess_and_split_data(currency_name, preprocess_func):
    """Preprocess data and split into train and test sets."""
    print(f"Processing {currency_name} data...")

    raw_data = load_data(os.path.join(PROCESSED_DATA_FOLDER, f"{currency_name}_processed.csv"))

    processed_data = preprocess_func(raw_data)

    
    X = processed_data.drop(columns=["target_close"])  
    y = processed_data["target_close"]  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train = scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test = scaler.transform(y_test.values.reshape(-1, 1))

    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_type, model_name):
    """Train a model, evaluate and save it to Artifacts."""
    print(f"Training model: {model_name} using {model_type}")

    if model_type == "linear_regression":
        model = LinearRegression()
    elif model_type == "xgboost":
        model = xgb.XGBRegressor(objective="reg:squarederror")
    else:
        raise ValueError("Invalid model type. Choose 'linear_regression' or 'xgboost'.")

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train) 
    y_pred_test = model.predict(X_test) 

 
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    mae_test = mean_absolute_error(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    print(f"{model_name} - Train MAE: {mae_train}, Train MSE: {mse_train}, Train R²: {r2_train}")
    print(f"{model_name} - Test MAE: {mae_test}, Test MSE: {mse_test}, Test R²: {r2_test}")

 
    model_filepath = os.path.join(ARTIFACTS_FOLDER, f"{model_name}.pkl")
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"{model_name} saved to {model_filepath}")

def main():
    try:
    
        print("Preparing Bitcoin data...")
        X_train_btc, X_test_btc, y_train_btc, y_test_btc = preprocess_and_split_data("bitcoin", preprocess_data)

 
        print("Preparing Ethereum data...")
        X_train_eth, X_test_eth, y_train_eth, y_test_eth = preprocess_and_split_data("ethereum", preprocess_data)

  
        print("Training and evaluating Linear Regression on Bitcoin data...")
        train_and_evaluate_model(X_train_btc, X_test_btc, y_train_btc, y_test_btc, "linear_regression", "linear_regression_btc")
        print("Training and evaluating XGBoost on Bitcoin data...")
        train_and_evaluate_model(X_train_btc, X_test_btc, y_train_btc, y_test_btc, "xgboost", "xgboost_btc")

        print("Training and evaluating Linear Regression on Ethereum data...")
        train_and_evaluate_model(X_train_eth, X_test_eth, y_train_eth, y_test_eth, "linear_regression", "linear_regression_eth")
        print("Training and evaluating XGBoost on Ethereum data...")
        train_and_evaluate_model(X_train_eth, X_test_eth, y_train_eth, y_test_eth, "xgboost", "xgboost_eth")

    except FileNotFoundError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
