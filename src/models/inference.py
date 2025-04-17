import requests
import pandas as pd
import pickle
import json
import os
from datetime import datetime, timezone

def fetch_realtime_data(symbol):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': '1d',
        'limit': 1
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()[0]

def prepare_features(kline):
    return pd.DataFrame([{
        'Open': float(kline[1]),
        'High': float(kline[2]),
        'Low': float(kline[3]),
        'Quote asset volume': float(kline[7]),
        'Number of trades': int(kline[8]),
        'Taker buy quote asset volume': float(kline[10]),
        'price_range': float(kline[2]) - float(kline[3])
    }])

def predict_next_close(symbol):
    # derive prefix from symbol
    prefix = symbol[:3].lower()  # 'btc' or 'eth'
    
    # load scaler and models
    scaler_path = os.path.join('artifacts', f'{prefix}_scaler.pkl')
    xgb_path    = os.path.join('artifacts', f'{prefix}_xgboost.pkl')
    lr_path     = os.path.join('artifacts', f'{prefix}_linear.pkl')

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(xgb_path, 'rb') as f:
        xgb = pickle.load(f)
    with open(lr_path, 'rb') as f:
        lr = pickle.load(f)

    # fetch and prepare
    kline = fetch_realtime_data(symbol)
    features = prepare_features(kline)
    
    # convert to numpy array to match how scaler was fitted
    features_array = features.values
    scaled_features = scaler.transform(features_array)

    # make predictions
    xgb_pred = float(xgb.predict(scaled_features)[0])
    lr_pred  = float(lr.predict(scaled_features)[0])
    timestamp = datetime.now(timezone.utc).isoformat()

    return {
        'XGBoost': xgb_pred,
        'LinearRegression': lr_pred,
        'Timestamp': timestamp
    }

def update_prediction_log(data, file_path='prediction.json'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            predictions = json.load(file)
    else:
        predictions = {}

    date_key = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    predictions[date_key] = data

    with open(file_path, 'w') as file:
        json.dump(predictions, file, indent=4)

def main():
    try:
        print("Fetching real-time data and making predictions...")

        btc = predict_next_close('BTCUSDT')
        eth = predict_next_close('ETHUSDT')

        print("\nBTC Next Day Close Prediction:")
        print(f"  XGBoost:           {btc['XGBoost']:.2f}")
        print(f"  Linear Regression: {btc['LinearRegression']:.2f}")

        print("\nETH Next Day Close Prediction:")
        print(f"  XGBoost:           {eth['XGBoost']:.2f}")
        print(f"  Linear Regression: {eth['LinearRegression']:.2f}")

        update_prediction_log({
            'BTC': {
                'XGBoost': btc['XGBoost'],
                'LinearRegression': btc['LinearRegression'],
                'Timestamp': btc['Timestamp']
            },
            'ETH': {
                'XGBoost': eth['XGBoost'],
                'LinearRegression': eth['LinearRegression'],
                'Timestamp': eth['Timestamp']
            }
        })

    except Exception as e:
        print(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
