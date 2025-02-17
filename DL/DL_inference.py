import os
import torch
import pandas as pd
import requests
import datetime
import numpy as np
from torch.utils.data import Dataset, DataLoader

ARTIFACTS_FOLDER = "../DL/artifacts"
LOG_FILE = "../DL/predictions.log"
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
INTERVAL = "1d"
PREDICTION_DAYS = 1
SEQ_LENGTH = 10

class CryptoDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

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

def load_model(model_name, model_class, input_size, hidden_size, num_layers, output_size):
    try:
        model_filepath = os.path.join(ARTIFACTS_FOLDER, f"{model_name}.pth")
        model = model_class(input_size, hidden_size, num_layers, output_size)
        model.load_state_dict(torch.load(model_filepath))
        model.eval()
        return model
    except FileNotFoundError:
        print(f"Error: Model file {model_name}.pth not found.")
        raise
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        raise

def fetch_binance_data(symbol, limit=100, start_time=None, end_time=None):
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
            "close": float(d[4]),
            "volume": float(d[5]),
            "quote_asset_volume": float(d[6]),
            "number_of_trades": int(float(d[7])),
            "taker_buy_base_asset_volume": float(d[8]),
            "taker_buy_quote_asset_volume": float(d[9]),
            "average_price": (float(d[2]) + float(d[3])) / 2,
            "price_change": float(d[4]) - float(d[1])
        } for d in data])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Binance API: {str(e)}")
        raise

def prepare_features(df):
    expected_features = [
        "open", "high", "low", "volume", "quote_asset_volume",
        "number_of_trades", "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume", "average_price", "price_change"
    ]
    return df[expected_features]

def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length].values)
    return np.array(X)

def generate_future_dates(start_date, periods):
    return [(start_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(1, periods + 1)]

def log_prediction(predictions):
    with open(LOG_FILE, "a") as f:
        for timestamp, symbol, prediction in predictions:
            f.write(f"{timestamp}, {symbol}, Predicted close: {prediction}\n")

def make_prediction(symbol, model_name, model_class, input_size, hidden_size, num_layers, output_size):
    try:
        model = load_model(model_name, model_class, input_size, hidden_size, num_layers, output_size)
        
        end_time = int(datetime.datetime.now().timestamp() * 1000)
        start_time = end_time - (PREDICTION_DAYS * 24 * 60 * 60 * 1000)
        
        input_data = fetch_binance_data(symbol, start_time=start_time, end_time=end_time)
        features = prepare_features(input_data)

        if features.empty:
            raise ValueError(f"No data available for {symbol}.")

        sequences = create_sequences(features, SEQ_LENGTH)
        sequences = torch.tensor(sequences, dtype=torch.float32)

        future_predictions = []
        for _ in range(PREDICTION_DAYS):
            with torch.no_grad():
                prediction = model(sequences[-1].unsqueeze(0)).item()
                future_predictions.append(prediction)

                new_row = np.append(sequences[-1].numpy()[1:], prediction).reshape(1, -1)
                sequences = torch.cat([sequences, torch.tensor(new_row, dtype=torch.float32).unsqueeze(0)])

        future_dates = generate_future_dates(datetime.datetime.now(), PREDICTION_DAYS)
        return list(zip(future_dates, [symbol] * PREDICTION_DAYS, future_predictions))

    except Exception as e:
        print(f"Error during prediction for {symbol}: {str(e)}")
        raise

if __name__ == "__main__":
    cryptos = [
        {"symbol": "BTCUSDT", "model": "btc_rnn_model", "model_class": RNNModel},
        {"symbol": "ETHUSDT", "model": "eth_rnn_model", "model_class": RNNModel},
        {"symbol": "BTCUSDT", "model": "btc_lstm_model", "model_class": LSTMModel},
        {"symbol": "ETHUSDT", "model": "eth_lstm_model", "model_class": LSTMModel}
    ]
    
    input_size = 10
    hidden_size = 64
    num_layers = 10
    output_size = 1

    try:
        all_predictions = []
        for crypto in cryptos:
            predictions = make_prediction(
                crypto["symbol"],
                crypto["model"],
                crypto["model_class"],
                input_size,
                hidden_size,
                num_layers,
                output_size
            )
            all_predictions.extend(predictions)
        
        log_prediction(all_predictions)
        for timestamp, symbol, predicted_close in all_predictions:
            print(f"Predicted close value for {symbol} on {timestamp}: {predicted_close}")
    except Exception as e:
        print(f"Error in prediction process: {str(e)}")