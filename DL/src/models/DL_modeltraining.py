import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import os
import pickle

PROJECT_DIR = r"D:\JMM_Technologies\CryptoCurrency-Forecasting\DL\data\processed_data"
ARTIFACTS_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "../artifacts")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def get_data(name):
    file_name = f"{name}.csv"
    file_path = os.path.join(PROJECT_DIR, file_name)
    return pd.read_csv(file_path)

bitcoin_df = get_data("bitcoin_processed")
ethereum_df = get_data("ethereum_processed")

scaler = MinMaxScaler()

def preprocess_data(df):
    features = ['open', 'high', 'low', 'volume', 'quote_asset_volume',
                'number_of_trades', 'taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume', 'average_price', 'price_change']
    target = 'target_close'
    
    df[features] = scaler.fit_transform(df[features])
    df[target] = scaler.fit_transform(df[[target]])
    
    return df, features, target

bitcoin_df, features, target = preprocess_data(bitcoin_df)
ethereum_df, _, _ = preprocess_data(ethereum_df)

def split_data(df, features, target):
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

X_btc_train, X_btc_test, y_btc_train, y_btc_test = split_data(bitcoin_df, features, target)
X_eth_train, X_eth_test, y_eth_train, y_eth_test = split_data(ethereum_df, features, target)

def train_lgbm(X_train, y_train):
    lgb_train = lgb.Dataset(X_train, label=y_train)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'verbose': -1
    }
    model = lgb.train(params, lgb_train, num_boost_round=100)
    return model

btc_lgbm_model = train_lgbm(X_btc_train, y_btc_train)
eth_lgbm_model = train_lgbm(X_eth_train, y_eth_train)

with open(os.path.join(ARTIFACTS_DIR, 'btc_lgbm_model.pkl'), 'wb') as f:
    pickle.dump(btc_lgbm_model, f)

with open(os.path.join(ARTIFACTS_DIR, 'eth_lgbm_model.pkl'), 'wb') as f:
    pickle.dump(eth_lgbm_model, f)

with open(os.path.join(ARTIFACTS_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return mse, mae, rmse

mse_btc_lgbm, mae_btc_lgbm, rmse_btc_lgbm = evaluate_model(btc_lgbm_model, X_btc_test, y_btc_test)
print(f'LightGBM Bitcoin - MSE: {mse_btc_lgbm:.6f}, MAE: {mae_btc_lgbm:.6f}, RMSE: {rmse_btc_lgbm:.6f}')

mse_eth_lgbm, mae_eth_lgbm, rmse_eth_lgbm = evaluate_model(eth_lgbm_model, X_eth_test, y_eth_test)
print(f'LightGBM Ethereum - MSE: {mse_eth_lgbm:.6f}, MAE: {mae_eth_lgbm:.6f}, RMSE: {rmse_eth_lgbm:.6f}')

print("==="*50)

def evaluate_model_with_inverse_scaling(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    
    y_test_actual = scaler.inverse_transform(y_test.values.reshape(-1, 1))
    predictions_actual = scaler.inverse_transform(predictions.reshape(-1, 1))
    
    mse = mean_squared_error(y_test_actual, predictions_actual)
    mae = mean_absolute_error(y_test_actual, predictions_actual)
    rmse = np.sqrt(mse)
    
    return mse, mae, rmse, y_test_actual, predictions_actual

mse_btc_lgbm_actual, mae_btc_lgbm_actual, rmse_btc_lgbm_actual, y_true_btc_lgbm, y_pred_btc_lgbm = evaluate_model_with_inverse_scaling(btc_lgbm_model, X_btc_test, y_btc_test, scaler)
print(f'LightGBM Bitcoin (Actual Values) - MSE: {mse_btc_lgbm_actual:.6f}, MAE: {mae_btc_lgbm_actual:.6f}, RMSE: {rmse_btc_lgbm_actual:.6f}')

mse_eth_lgbm_actual, mae_eth_lgbm_actual, rmse_eth_lgbm_actual, y_true_eth_lgbm, y_pred_eth_lgbm = evaluate_model_with_inverse_scaling(eth_lgbm_model, X_eth_test, y_eth_test, scaler)
print(f'LightGBM Ethereum (Actual Values) - MSE: {mse_eth_lgbm_actual:.6f}, MAE: {mae_eth_lgbm_actual:.6f}, RMSE: {rmse_eth_lgbm_actual:.6f}')

print("==="*50)

class CryptoDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, features, target, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[features].iloc[i:i+seq_length].values)
        y.append(data[target].iloc[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 10
X_btc_train_rnn, y_btc_train_rnn = create_sequences(bitcoin_df, features, target, SEQ_LENGTH)
X_btc_test_rnn, y_btc_test_rnn = create_sequences(bitcoin_df, features, target, SEQ_LENGTH)
X_eth_train_rnn, y_eth_train_rnn = create_sequences(ethereum_df, features, target, SEQ_LENGTH)
X_eth_test_rnn, y_eth_test_rnn = create_sequences(ethereum_df, features, target, SEQ_LENGTH)

X_btc_train_rnn, y_btc_train_rnn = torch.tensor(X_btc_train_rnn, dtype=torch.float32), torch.tensor(y_btc_train_rnn, dtype=torch.float32)
X_btc_test_rnn, y_btc_test_rnn = torch.tensor(X_btc_test_rnn, dtype=torch.float32), torch.tensor(y_btc_test_rnn, dtype=torch.float32)
X_eth_train_rnn, y_eth_train_rnn = torch.tensor(X_eth_train_rnn, dtype=torch.float32), torch.tensor(y_eth_train_rnn, dtype=torch.float32)
X_eth_test_rnn, y_eth_test_rnn = torch.tensor(X_eth_test_rnn, dtype=torch.float32), torch.tensor(y_eth_test_rnn, dtype=torch.float32)

batch_size = 64
btc_train_loader_rnn = DataLoader(CryptoDataset(X_btc_train_rnn, y_btc_train_rnn), batch_size=batch_size, shuffle=False)
btc_test_loader_rnn = DataLoader(CryptoDataset(X_btc_test_rnn, y_btc_test_rnn), batch_size=batch_size, shuffle=False)
eth_train_loader_rnn = DataLoader(CryptoDataset(X_eth_train_rnn, y_eth_train_rnn), batch_size=batch_size, shuffle=False)
eth_test_loader_rnn = DataLoader(CryptoDataset(X_eth_test_rnn, y_eth_test_rnn), batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = len(features)
hidden_size = 64
num_layers = 10
output_size = 1
learning_rate = 0.001
num_epochs = 200

def train_model(model, train_loader, criterion, optimizer):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.6f}')

btc_rnn_model = RNNModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(btc_rnn_model.parameters(), lr=learning_rate)
train_model(btc_rnn_model, btc_train_loader_rnn, criterion, optimizer)

eth_rnn_model = RNNModel(input_size, hidden_size, num_layers, output_size).to(device)
optimizer = optim.Adam(eth_rnn_model.parameters(), lr=learning_rate)
train_model(eth_rnn_model, eth_train_loader_rnn, criterion, optimizer)

torch.save(btc_rnn_model.state_dict(), os.path.join(ARTIFACTS_DIR, 'btc_rnn_model.pth'))
torch.save(eth_rnn_model.state_dict(), os.path.join(ARTIFACTS_DIR, 'eth_rnn_model.pth'))

def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        X, y = X.to(device), y.to(device)
        predictions = model(X).squeeze().cpu().numpy()
        y_true = y.cpu().numpy()
        mse = mean_squared_error(y_true, predictions)
        mae = mean_absolute_error(y_true, predictions)
        rmse = np.sqrt(mse)
        return mse, mae, rmse

mse_btc_rnn, mae_btc_rnn, rmse_btc_rnn = evaluate_model(btc_rnn_model, X_btc_test_rnn, y_btc_test_rnn)
print(f'RNN Bitcoin - MSE: {mse_btc_rnn:.6f}, MAE: {mae_btc_rnn:.6f}, RMSE: {rmse_btc_rnn:.6f}')

mse_eth_rnn, mae_eth_rnn, rmse_eth_rnn = evaluate_model(eth_rnn_model, X_eth_test_rnn, y_eth_test_rnn)
print(f'RNN Ethereum - MSE: {mse_eth_rnn:.6f}, MAE: {mae_eth_rnn:.6f}, RMSE: {rmse_eth_rnn:.6f}')

def evaluate_model_with_inverse_scaling(model, X, y, scaler):
    model.eval()
    with torch.no_grad():
        X, y = X.to(device), y.to(device)
        predictions = model(X).squeeze().cpu().numpy()
        y_true = y.cpu().numpy()

        y_true = y_true.reshape(-1, 1)
        predictions = predictions.reshape(-1, 1)

        y_true_actual = scaler.inverse_transform(y_true)
        predictions_actual = scaler.inverse_transform(predictions)

        mse = mean_squared_error(y_true_actual, predictions_actual)
        mae = mean_absolute_error(y_true_actual, predictions_actual)
        rmse = np.sqrt(mse)

        return mse, mae, rmse, y_true_actual, predictions_actual

mse_btc_rnn_actual, mae_btc_rnn_actual, rmse_btc_rnn_actual, y_true_btc_rnn, y_pred_btc_rnn = evaluate_model_with_inverse_scaling(btc_rnn_model, X_btc_test_rnn, y_btc_test_rnn, scaler)
print(f'RNN Bitcoin (Actual Values) - MSE: {mse_btc_rnn_actual:.6f}, MAE: {mae_btc_rnn_actual:.6f}, RMSE: {rmse_btc_rnn_actual:.6f}')

mse_eth_rnn_actual, mae_eth_rnn_actual, rmse_eth_rnn_actual, y_true_eth_rnn, y_pred_eth_rnn = evaluate_model_with_inverse_scaling(eth_rnn_model, X_eth_test_rnn, y_eth_test_rnn, scaler)
print(f'RNN Ethereum (Actual Values) - MSE: {mse_eth_rnn_actual:.6f}, MAE: {mae_eth_rnn_actual:.6f}, RMSE: {rmse_eth_rnn_actual:.6f}')

print("==="*50)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

btc_lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(btc_lstm_model.parameters(), lr=learning_rate)
train_model(btc_lstm_model, btc_train_loader_rnn, criterion, optimizer)

eth_lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
optimizer = optim.Adam(eth_lstm_model.parameters(), lr=learning_rate)
train_model(eth_lstm_model, eth_train_loader_rnn, criterion, optimizer)

# Save LSTM models
torch.save(btc_lstm_model.state_dict(), os.path.join(ARTIFACTS_DIR, 'btc_lstm_model.pth'))
torch.save(eth_lstm_model.state_dict(), os.path.join(ARTIFACTS_DIR, 'eth_lstm_model.pth'))

mse_btc_lstm, mae_btc_lstm, rmse_btc_lstm = evaluate_model(btc_lstm_model, X_btc_test_rnn, y_btc_test_rnn)
print(f'LSTM Bitcoin - MSE: {mse_btc_lstm:.6f}, MAE: {mae_btc_lstm:.6f}, RMSE: {rmse_btc_lstm:.6f}')

mse_eth_lstm, mae_eth_lstm, rmse_eth_lstm = evaluate_model(eth_lstm_model, X_eth_test_rnn, y_eth_test_rnn)
print(f'LSTM Ethereum - MSE: {mse_eth_lstm:.6f}, MAE: {mae_eth_lstm:.6f}, RMSE: {rmse_eth_lstm:.6f}')

def evaluate_model_with_inverse_scaling(model, X, y, scaler):
    model.eval()
    with torch.no_grad():
        X, y = X.to(device), y.to(device)
        predictions = model(X).squeeze().cpu().numpy()
        y_true = y.cpu().numpy()

        y_true = y_true.reshape(-1, 1)
        predictions = predictions.reshape(-1, 1)

        y_true_actual = scaler.inverse_transform(y_true)
        predictions_actual = scaler.inverse_transform(predictions)

        mse = mean_squared_error(y_true_actual, predictions_actual)
        mae = mean_absolute_error(y_true_actual, predictions_actual)
        rmse = np.sqrt(mse)

        return mse, mae, rmse, y_true_actual, predictions_actual

mse_btc_lstm_actual, mae_btc_lstm_actual, rmse_btc_lstm_actual, y_true_btc_lstm, y_pred_btc_lstm = evaluate_model_with_inverse_scaling(btc_lstm_model, X_btc_test_rnn, y_btc_test_rnn, scaler)
print(f'LSTM Bitcoin (Actual Values) - MSE: {mse_btc_lstm_actual:.6f}, MAE: {mae_btc_lstm_actual:.6f}, RMSE: {rmse_btc_lstm_actual:.6f}')

mse_eth_lstm_actual, mae_eth_lstm_actual, rmse_eth_lstm_actual, y_true_eth_lstm, y_pred_eth_lstm = evaluate_model_with_inverse_scaling(eth_lstm_model, X_eth_test_rnn, y_eth_test_rnn, scaler)
print(f'LSTM Ethereum (Actual Values) - MSE: {mse_eth_lstm_actual:.6f}, MAE: {mae_eth_lstm_actual:.6f}, RMSE: {rmse_eth_lstm_actual:.6f}')

print("==="*50)

results = {
    'LightGBM Bitcoin': {
        'MSE': mse_btc_lgbm,
        'MAE': mae_btc_lgbm,
        'RMSE': rmse_btc_lgbm,
        'MSE (Actual Values)': mse_btc_lgbm_actual,
        'MAE (Actual Values)': mae_btc_lgbm_actual,
        'RMSE (Actual Values)': rmse_btc_lgbm_actual
    },
    'LightGBM Ethereum': {
        'MSE': mse_eth_lgbm,
        'MAE': mae_eth_lgbm,
        'RMSE': rmse_eth_lgbm,
        'MSE (Actual Values)': mse_eth_lgbm_actual,
        'MAE (Actual Values)': mae_eth_lgbm_actual,
        'RMSE (Actual Values)': rmse_eth_lgbm_actual
    },
    'RNN Bitcoin': {
        'MSE': mse_btc_rnn,
        'MAE': mae_btc_rnn,
        'RMSE': rmse_btc_rnn,
        'MSE (Actual Values)': mse_btc_rnn_actual,
        'MAE (Actual Values)': mae_btc_rnn_actual,
        'RMSE (Actual Values)': rmse_btc_rnn_actual
    },
    'RNN Ethereum': {
        'MSE': mse_eth_rnn,
        'MAE': mae_eth_rnn,
        'RMSE': rmse_eth_rnn,
        'MSE (Actual Values)': mse_eth_rnn_actual,
        'MAE (Actual Values)': mae_eth_rnn_actual,
        'RMSE (Actual Values)': rmse_eth_rnn_actual
    },
    'LSTM Bitcoin': {
        'MSE': mse_btc_lstm,
        'MAE': mae_btc_lstm,
        'RMSE': rmse_btc_lstm,
        'MSE (Actual Values)': mse_btc_lstm_actual,
        'MAE (Actual Values)': mae_btc_lstm_actual,
        'RMSE (Actual Values)': rmse_btc_lstm_actual
    },
    'LSTM Ethereum': {
        'MSE': mse_eth_lstm,
        'MAE': mae_eth_lstm,
        'RMSE': rmse_eth_lstm,
        'MSE (Actual Values)': mse_eth_lstm_actual,
        'MAE (Actual Values)': mae_eth_lstm_actual,
        'RMSE (Actual Values)': rmse_eth_lstm_actual
    }
}

import json
results_file_path = os.path.join(ARTIFACTS_DIR, 'evaluation_results.json')
with open(results_file_path, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Evaluation results saved to {results_file_path}")