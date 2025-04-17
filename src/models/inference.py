import requests
import pandas as pd
import pickle

def fetch_realtime_data(symbol):
    """Fetch real-time daily candle data from Binance API"""
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
    """Prepare features matching the model's training format"""
    features = {
        'Open': float(kline[1]),
        'High': float(kline[2]),
        'Low': float(kline[3]),
        'Quote asset volume': float(kline[7]),
        'Number of trades': int(kline[8]),
        'Taker buy quote asset volume': float(kline[10]),
        'price_range': float(kline[2]) - float(kline[3])
    }
    return pd.DataFrame([features])

def predict_next_close(symbol):
    """Make predictions using both models for a given symbol"""
    prefix = 'btc' if symbol == 'BTCUSDT' else 'eth'
    
    # Load artifacts
    scaler = pickle.load(open(f'../../artifacts/{prefix}_scaler.pkl', 'rb'))
    xgb = pickle.load(open(f'../../artifacts/{prefix}_xgboost.pkl', 'rb'))
    lr = pickle.load(open(f'../../artifacts/{prefix}_linear.pkl', 'rb'))

    # Get and prepare data
    kline = fetch_realtime_data(symbol)
    features = prepare_features(kline)
    
    # Scale features
    scaled_features = scaler.transform(features)
    
    return {
        'XGBoost': xgb.predict(scaled_features)[0],
        'LinearRegression': lr.predict(scaled_features)[0]
    }

def main():
    """Main prediction routine"""
    try:
        print("Fetching real-time data and making predictions...")
        
        # Bitcoin prediction
        btc = predict_next_close('BTCUSDT')
        print("\nBTC Next Day Close Prediction:")
        print(f"XGBoost: {btc['XGBoost']:.2f}")
        print(f"Linear Regression: {btc['LinearRegression']:.2f}")

        # Ethereum prediction
        eth = predict_next_close('ETHUSDT')
        print("\nETH Next Day Close Prediction:")
        print(f"XGBoost: {eth['XGBoost']:.2f}")
        print(f"Linear Regression: {eth['LinearRegression']:.2f}")

    except Exception as e:
        print(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()