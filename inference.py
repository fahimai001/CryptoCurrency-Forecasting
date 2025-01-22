import os
import sys
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

# Get the absolute path to the project root and add the 'SFC' folder to the path
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the absolute path of the current script
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))  # Go up two levels to the project root

# Add 'SFC' directory to the path (from project root)
sys.path.append(os.path.join(project_root, 'SFC'))

# Now you can import your modules
from prediction_pipeline.btc_data_ingestion_and_preprocessing import preprocess_data as preprocess_btc
from prediction_pipeline.eth_data_ingestion_and_preprocessing import preprocess_data as preprocess_eth

# Define the paths
ARTIFACTS_FOLDER = os.path.join(project_root, 'artifacts')

# Load saved models and scalers
def load_model_and_scalers(currency_name):
    """Load the saved model and scalers for the given currency."""
    # Load scalers
    with open(os.path.join(ARTIFACTS_FOLDER, f"{currency_name}_feature_scaler.pkl"), "rb") as f:
        feature_scaler = pickle.load(f)
    
    with open(os.path.join(ARTIFACTS_FOLDER, f"{currency_name}_target_scaler.pkl"), "rb") as f:
        target_scaler = pickle.load(f)
    
    # Load models
    with open(os.path.join(ARTIFACTS_FOLDER, f"linear_regression_{currency_name}.pkl"), "rb") as f:
        lr_model = pickle.load(f)
    
    with open(os.path.join(ARTIFACTS_FOLDER, f"xgboost_{currency_name}.pkl"), "rb") as f:
        xgb_model = pickle.load(f)
    
    return lr_model, xgb_model, feature_scaler, target_scaler

def make_prediction(currency_name, input_data):
    """Make predictions using the trained models and scalers."""
    # Preprocess the input data (assuming it's a DataFrame)
    if currency_name == "bitcoin":
        processed_data = preprocess_btc(input_data)
    elif currency_name == "ethereum":
        processed_data = preprocess_eth(input_data)
    else:
        raise ValueError("Invalid currency name. Choose 'bitcoin' or 'ethereum'.")
    
    # Select features (similar to the model training step)
    X = processed_data.drop(columns=["close_time", "timestamp", "close"], errors="ignore")

    # Load model and scalers
    lr_model, xgb_model, feature_scaler, target_scaler = load_model_and_scalers(currency_name)

    # Scale the features using the saved scaler
    X_scaled = feature_scaler.transform(X)

    # Predict using both models
    lr_prediction = lr_model.predict(X_scaled)
    xgb_prediction = xgb_model.predict(X_scaled)

    # Reverse scale the predictions to get the original scale
    lr_prediction = target_scaler.inverse_transform(lr_prediction.reshape(-1, 1))
    xgb_prediction = target_scaler.inverse_transform(xgb_prediction.reshape(-1, 1))

    return lr_prediction, xgb_prediction

def main():
    # Example input data for prediction (replace with actual input data)
    input_data = pd.DataFrame({
        # Include the same features as in your processed data (excluding 'close_time', 'timestamp', 'close')
        "feature_1": [0.5],  # example values
        "feature_2": [1.2],  # example values
        # Add more features as required by your model
    })
    
    # Make predictions for Bitcoin
    lr_prediction_btc, xgb_prediction_btc = make_prediction("bitcoin", input_data)
    print(f"Linear Regression Prediction for Bitcoin: {lr_prediction_btc}")
    print(f"XGBoost Prediction for Bitcoin: {xgb_prediction_btc}")
    
    # Make predictions for Ethereum
    lr_prediction_eth, xgb_prediction_eth = make_prediction("ethereum", input_data)
    print(f"Linear Regression Prediction for Ethereum: {lr_prediction_eth}")
    print(f"XGBoost Prediction for Ethereum: {xgb_prediction_eth}")

if __name__ == "__main__":
    main()
