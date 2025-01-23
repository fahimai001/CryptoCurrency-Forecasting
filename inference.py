import os
import pickle
import pandas as pd

# Define the paths to artifacts
ARTIFACTS_FOLDER = "./artifacts"

# Load a saved model
def load_model(model_name):
    model_filepath = os.path.join(ARTIFACTS_FOLDER, f"{model_name}.pkl")
    if not os.path.exists(model_filepath):
        raise FileNotFoundError(f"The model {model_name} does not exist.")
    with open(model_filepath, "rb") as f:
        model = pickle.load(f)
    return model

# Load a saved scaler
def load_scaler(scaler_name):
    scaler_filepath = os.path.join(ARTIFACTS_FOLDER, f"{scaler_name}.pkl")
    if not os.path.exists(scaler_filepath):
        raise FileNotFoundError(f"The scaler {scaler_name} does not exist.")
    with open(scaler_filepath, "rb") as f:
        scaler = pickle.load(f)
    return scaler

# Predict using the model
def make_prediction(input_data, model_name, feature_scaler_name, target_scaler_name):
    # Load model and scalers
    model = load_model(model_name)
    feature_scaler = load_scaler(feature_scaler_name)
    target_scaler = load_scaler(target_scaler_name)
    
    # Scale the input data
    input_scaled = feature_scaler.transform(input_data)
    
    # Make predictions
    predictions_scaled = model.predict(input_scaled)
    
    # Inverse scale the predictions
    predictions = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1))
    return predictions

# Example usage
if __name__ == "__main__":
    try:
        input_data = pd.DataFrame({
            "open": [99781.89], 
            "high": [100786.3],
            "low": [99608.23],
            "volume": [2207.59419],
            "quote_asset_volume": [221303048.919697],
            "number_of_trades": [472723],
            "taker_buy_base_asset_volume": [1125.00115],
            "taker_buy_quote_asset_volume": [112761937.89211]
        })
        
        # Specify model and scaler names
        model_name = "xgboost_btc"  # Change to your model's name
        feature_scaler_name = "bitcoin_feature_scaler"
        target_scaler_name = "bitcoin_target_scaler"
        
        # Make predictions
        predictions = make_prediction(input_data, model_name, feature_scaler_name, target_scaler_name)
        
        print(f"Predicted 'close' value: {predictions[0][0]}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
