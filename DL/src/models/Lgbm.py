import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
from sklearn.preprocessing import MinMaxScaler

PROJECT_DIR = r"D:\JMM_Technologies\CryptoCurrency-Forecasting\DL\data\processed_data"

def get_data(name):
    """Load dataset by name."""
    file_name = f"{name}.csv"
    file_path = os.path.join(PROJECT_DIR, file_name)
    return pd.read_csv(file_path)

def prepare_data(df):
    """Prepares data by splitting into train/test and applying MinMax scaling."""
    df = df.dropna()
    X = df.drop(columns=["target_close"])
    y = df["target_close"]
    
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_scaler, y_scaler

def train_lgbm(X_train, y_train, X_test, y_test, y_scaler, dataset_name):
    """Trains an LGBM model and evaluates its performance."""
    train_data = lgb.Dataset(X_train, label=y_train)
    
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1
    }
    
    model = lgb.train(params, train_data, num_boost_round=1000)
    y_pred_scaled = model.predict(X_test)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_original = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    mse = mean_squared_error(y_test_original, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, y_pred)
    
    print(f"Metrics for {dataset_name}:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    return model

def tune_lgbm(X_train, y_train):
    """Tunes the LGBM model using RandomizedSearchCV."""
    model = lgb.LGBMRegressor(objective="regression", metric="rmse", verbose=-1)
    
    param_grid = {
        "num_leaves": [31, 50, 100],
        "learning_rate": [0.01, 0.05, 0.1],
        "feature_fraction": [0.8, 0.9, 1.0],
        "bagging_fraction": [0.8, 0.9, 1.0],
        "min_gain_to_split": [0.0, 0.01, 0.1, 1],
        "bagging_freq": [5, 10, 20]
    }
    
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_iter=10,
        verbose=1,
        random_state=42
    )
    random_search.fit(X_train, y_train)
    
    print(f"Best parameters: {random_search.best_params_}")
    return random_search.best_estimator_

btc_df = get_data("bitcoin_processed")
eth_df = get_data("ethereum_processed")

btc_X_train, btc_X_test, btc_y_train, btc_y_test, btc_X_scaler, btc_y_scaler = prepare_data(btc_df)
eth_X_train, eth_X_test, eth_y_train, eth_y_test, eth_X_scaler, eth_y_scaler = prepare_data(eth_df)

btc_model = train_lgbm(btc_X_train, btc_y_train, btc_X_test, btc_y_test, btc_y_scaler, "Bitcoin")
btc_tuned_model = tune_lgbm(btc_X_train, btc_y_train)
eth_model = train_lgbm(eth_X_train, eth_y_train, eth_X_test, eth_y_test, eth_y_scaler, "Ethereum")
eth_tuned_model = tune_lgbm(eth_X_train, eth_y_train)

if not os.path.exists("../artifacts"):
    os.makedirs("../artifacts")

joblib.dump(btc_model, "../artifacts/btc_lgbm_model.pkl")
joblib.dump(btc_tuned_model, "../artifacts/btc_lgbm_tuned_model.pkl")
joblib.dump(eth_model, "../artifacts/eth_lgbm_model.pkl")
joblib.dump(eth_tuned_model, "../artifacts/eth_lgbm_tuned_model.pkl")