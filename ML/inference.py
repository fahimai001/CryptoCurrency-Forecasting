import os
import pickle
import pandas as pd
import requests
import datetime
import numpy as np

# Define constants
ARTIFACTS_FOLDER = "../ML/artifacts"
LOG_FILE = "../ML/predictions.log"
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
INTERVAL = "1d"
PREDICTION_DAYS = 1

# Load a saved model
def load_model(model_name):
    try:
        model_filepath = os.path.join(ARTIFACTS_FOLDER, f"{model_name}.pkl")
        with open(model_filepath, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Model file {model_name}.pkl not found.")
        raise
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        raise

# Fetch historical data from Binance API
def fetch_binance_data(symbol, limit=50, start_time=None, end_time=None):
    try:
        params = {"symbol": symbol, "interval": INTERVAL, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        response = requests.get(BINANCE_API_URL, params=params)
        response.raise_for_status()
        data = response.json()

        return pd.DataFrame([{
            "open": float(d[1]),
            "high": float(d[2]),
            "low": float(d[3]),
            "volume": float(d[4]),
            "quote_asset_volume": float(d[5]),
            "number_of_trades": int(float(d[6])),
            "taker_buy_base_asset_volume": float(d[7]),
            "taker_buy_quote_asset_volume": float(d[8]),
            "average_price": float(d[9]),
            "price_change": float(d[10])
        } for d in data])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Binance API: {str(e)}")
        raise

# Ensure feature consistency
def prepare_features(df):
    expected_features = df.columns
    return df[expected_features]

# Generate future timestamps
def generate_future_dates(start_date, periods):
    return [(start_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(1, periods + 1)]

# Log predictions
def log_prediction(predictions):
    with open(LOG_FILE, "a") as f:
        for timestamp, symbol, prediction in predictions:
            f.write(f"{timestamp}, {symbol}, Predicted close: {prediction}\n")

# Make prediction
def make_prediction(symbol, model_name):
    try:
        model = load_model(model_name)
        
        # Calculate start_time and end_time for the most recent data
        end_time = int(datetime.datetime.now().timestamp() * 1000)
        start_time = end_time - (PREDICTION_DAYS * 24 * 60 * 60 * 1000)
        
        input_data = fetch_binance_data(symbol, start_time=start_time, end_time=end_time)
        features = prepare_features(input_data)

        if features.empty:
            raise ValueError(f"No data available for {symbol}.")

        future_predictions = []
        for _ in range(PREDICTION_DAYS):
            last_row = features.iloc[-1].values.reshape(1, -1)
            prediction = model.predict(last_row)[0]
            future_predictions.append(prediction)

            new_row = np.append(last_row[0][1:], prediction).reshape(1, -1)
            features = pd.DataFrame(np.vstack([features.values, new_row]), columns=features.columns)

        future_dates = generate_future_dates(datetime.datetime.now(), PREDICTION_DAYS)
        return list(zip(future_dates, [symbol] * PREDICTION_DAYS, future_predictions))

    except Exception as e:
        print(f"Error during prediction for {symbol}: {str(e)}")
        raise

if __name__ == "__main__":
    cryptos = [
        {"symbol": "BTCUSDT", "model": "xgboost_btc"},
        {"symbol": "ETHUSDT", "model": "xgboost_eth"}
    ]
    
    try:
        all_predictions = []
        for crypto in cryptos:
            predictions = make_prediction(crypto["symbol"], crypto["model"])
            all_predictions.extend(predictions)
        
        log_prediction(all_predictions)
        for timestamp, symbol, predicted_close in all_predictions:
            print(f"Predicted close value for {symbol} on {timestamp}: {predicted_close}")
    except Exception as e:
        print(f"Error in prediction process: {str(e)}")