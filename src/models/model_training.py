import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import uniform, randint
import pickle

# Ensure artifacts directory exists
def ensure_artifacts_dir():
    os.makedirs("artifacts", exist_ok=True)


def evaluate_model(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name:<15} MSE:  {mse:.4f}")
    print(f"{name:<15} RMSE: {rmse:.4f}")
    print(f"{name:<15} MAE:  {mae:.4f}")
    print(f"{name:<15} R2:   {r2:.4f}")
    print("-" * 60)
    return rmse

def train_and_evaluate(name, csv_path):
    print(f"\n{'='*10} {name} Dataset {'='*10}\n")
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["target"]).values
    y = df["target"].values

    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    xgb_base = XGBRegressor(objective='reg:squarederror', random_state=42, tree_method='hist')
    param_distributions = {
        'n_estimators': randint(100, 1000),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_weight': randint(1, 10),
        'gamma': uniform(0, 0.5),
        'reg_alpha': uniform(0, 2),
        'reg_lambda': uniform(0.5, 2)
    }
    xgb_rs = RandomizedSearchCV(
        xgb_base,
        param_distributions=param_distributions,
        n_iter=30,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    xgb_rs.fit(X_train_s, y_train)

    best_params = xgb_rs.best_params_
    print("Best XGB params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"Best XGB CV MSE: {-xgb_rs.best_score_:.4f}")
    print("-" * 60)

    xgb_final = XGBRegressor(**best_params, objective='reg:squarederror', random_state=42)
    xgb_final.fit(X_train_s, y_train)
    lr = LinearRegression().fit(X_train_s, y_train)

    xgb_pred = xgb_final.predict(X_test_s)
    lr_pred = lr.predict(X_test_s)

    rmse_xgb = evaluate_model("XGBoost", y_test, xgb_pred)
    rmse_lr = evaluate_model("LinearRegression", y_test, lr_pred)

    all_X_s = np.vstack([X_train_s, X_test_s])
    all_y = np.concatenate([y_train, y_test])
    xgb_cv_rmse = np.sqrt(-cross_val_score(
        xgb_final,
        all_X_s, all_y,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    ).mean())
    print(f"XGBoost 5‑fold CV RMSE: {xgb_cv_rmse:.4f}")
    print("\n" + "="*50 + "\n")

    prefix = 'btc' if name.lower() == 'bitcoin' else 'eth'
    pickle.dump(xgb_final, open(f"artifacts/{prefix}_xgboost.pkl", "wb"))
    pickle.dump(lr, open(f"artifacts/{prefix}_linear.pkl", "wb"))
    pickle.dump(scaler, open(f"artifacts/{prefix}_scaler.pkl", "wb"))

    return {
        "XGB_RMSE": rmse_xgb,
        "LR_RMSE": rmse_lr,
        "XGB_CV_RMSE": xgb_cv_rmse
    }

if __name__ == "__main__":
    ensure_artifacts_dir()
    datasets = {
        "Bitcoin": "data/processed/final_bitcoin.csv",
        "Ethereum": "data/processed/final_ethereum.csv"
    }
    summary = {}
    for name, path in datasets.items():
        summary[name] = train_and_evaluate(name, path)
    summary_df = pd.DataFrame(summary).T
    print("Summary of Test RMSE and 5‑fold CV RMSE:\n")
    print(summary_df.to_string(float_format="%.4f"))
