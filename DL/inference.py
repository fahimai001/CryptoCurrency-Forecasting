import os
import pickle
import pandas as pd
import requests
import datetime
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ARTIFACTS_FOLDER = "../DL/artifacts"
LOG_FILE = "../DL/predictions.log"
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
INTERVAL = "1d"
PREDICTION_DAYS = 1

# Define PyTorch models (RNN and LSTM)
class RNNModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def load_model(model_name):
    try:
        if "lgbm" in model_name:
            model_filepath = os.path.join(ARTIFACTS_FOLDER, f"{model_name}.pkl")
            with open(model_filepath, "rb") as f:
                return pickle.load(f)
        elif "rnn" in model_name or "lstm" in model_name:
            model_filepath = os.path.join(ARTIFACTS_FOLDER, f"{model_name}.pth")
            if "rnn" in model_name:
                model = RNNModel(input_size=10, hidden_size=64, num_layers=10, output_size=1)
            else:
                model = LSTMModel(input_size=10, hidden_size=64, num_layers=10, output_size=1)
            model.load_state_dict(torch.load(model_filepath))
            model.eval()
            return model
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
    except FileNotFoundError:
        print(f"Error: Model file {model_name} not found.")
        raise
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        raise

def load_scaler():
    try:
        scaler_filepath = os.path.join(ARTIFACTS_FOLDER, "scaler.pkl")
        with open(scaler_filepath, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("Error: Scaler file not found.")
        raise
    except Exception as e:
        print(f"Error loading scaler: {str(e)}")
        raise

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

def prepare_features(df, scaler):
    expected_features = ['open', 'high', 'low', 'volume', 'quote_asset_volume',
                        'number_of_trades', 'taker_buy_base_asset_volume',
                        'taker_buy_quote_asset_volume', 'average_price', 'price_change']
    df[expected_features] = scaler.transform(df[expected_features])
    return df[expected_features]

def generate_future_dates(start_date, periods):
    return [(start_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(1, periods + 1)]

def log_prediction(predictions):
    with open(LOG_FILE, "a") as f:
        for timestamp, symbol, model_name, prediction in predictions:
            f.write(f"{timestamp}, {symbol}, {model_name}, Predicted close: {prediction}\n")

def make_prediction(symbol, model_name, scaler):
    try:
        model = load_model(model_name)
        
        end_time = int(datetime.datetime.now().timestamp() * 1000)
        start_time = end_time - (PREDICTION_DAYS * 24 * 60 * 60 * 1000)
        
        input_data = fetch_binance_data(symbol, start_time=start_time, end_time=end_time)
        features = prepare_features(input_data, scaler)

        if features.empty:
            raise ValueError(f"No data available for {symbol}.")

        future_predictions = []
        for _ in range(PREDICTION_DAYS):
            if "lgbm" in model_name:
                last_row = features.iloc[-1].values.reshape(1, -1)
                prediction = model.predict(last_row)[0]
            elif "rnn" in model_name or "lstm" in model_name:
                last_row = torch.tensor(features.iloc[-1].values.reshape(1, -1), dtype=torch.float32)
                prediction = model(last_row).item()
            else:
                raise ValueError(f"Unsupported model type: {model_name}")

            future_predictions.append(prediction)

            new_row = np.append(features.iloc[-1].values[1:], prediction).reshape(1, -1)
            features = pd.DataFrame(np.vstack([features.values, new_row]), columns=features.columns)

        future_dates = generate_future_dates(datetime.datetime.now(), PREDICTION_DAYS)
        return list(zip(future_dates, [symbol] * PREDICTION_DAYS, [model_name] * PREDICTION_DAYS, future_predictions))

    except Exception as e:
        print(f"Error during prediction for {symbol} using {model_name}: {str(e)}")
        raise

if __name__ == "__main__":
    cryptos = [
        {"symbol": "BTCUSDT", "models": ["btc_lgbm_model", "btc_rnn_model", "btc_lstm_model"]},
        {"symbol": "ETHUSDT", "models": ["eth_lgbm_model", "eth_rnn_model", "eth_lstm_model"]}
    ]
    
    try:
        scaler = load_scaler()
        all_predictions = []
        for crypto in cryptos:
            for model_name in crypto["models"]:
                predictions = make_prediction(crypto["symbol"], model_name, scaler)
                all_predictions.extend(predictions)
        
        log_prediction(all_predictions)
        for timestamp, symbol, model_name, predicted_close in all_predictions:
            print(f"Predicted close value for {symbol} using {model_name} on {timestamp}: {predicted_close}")
    except Exception as e:
        print(f"Error in prediction process: {str(e)}")