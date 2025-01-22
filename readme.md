crypto_forecast
├── artifacts
│   ├── bitcoin_feature_scaler.pkl
│   ├── bitcoin_target_scaler.pkl
│   ├── ethereum_feature_scaler.pkl
│   ├── ethereum_target_scaler.pkl
│   ├── linear_regression_btc.pkl
│   ├── linear_regression_eth.pkl
│   ├── xgboost_btc.pkl
│   └── xgboost_eth.pkl
├── crypto_forecast
│   ├── include
│   │   ├── Lib
│   │   │   └── ... (scripts)
│   │   └── Scripts
│   │       └── ... (scripts)
├── data
│   ├── processed_data
│   │   ├── bitcoin_processed.csv
│   │   └── ethereum_processed.csv
│   └── raw_data
│       ├── bitcoin.csv
│       └── ethereum.csv
├── notebook
│   └── experiment.ipynb
├── SFC
│   ├── data
│   │   └── Data_Preprocessing
│   │       ├── __pycache__
│   │       ├── init.py
│   │       └── data_ingestion.py
│   ├── models
│   │   ├── __pycache__
│   │   ├── init.py
│   │   ├── model_testing.py
│   │   └── modeltraining.py
│   └── prediction_pipeline
│       ├── __pycache__
│       ├── init.py
│       ├── btc_data_ingestion_and_preprocessing.py
│       └── eth_data_ingestion_and_preprocessing.py
├── init.py
├── gitignore
├── inference.py
├── readme.md
└── requirements.txt