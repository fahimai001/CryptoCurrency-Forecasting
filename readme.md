# Cryptocurrency Forecasting

This project aims to predict the **close price** of Bitcoin and Ethereum using machine learning models. The system ingests raw data, preprocesses it, trains various models, and provides predictions for cryptocurrency prices. It is built with Python and incorporates data preprocessing, feature scaling, and ML-based forecasting.

---

## Project Structure

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


---

## Features

- **Data Ingestion**: Reads raw cryptocurrency data (Bitcoin and Ethereum) for preprocessing.
- **Preprocessing**: Scales features and targets for effective model training.
- **Model Training**: Implements and trains machine learning models (Linear Regression, XGBoost).
- **Model Evaluation**: Evaluates model performance using metrics.
- **Prediction Pipeline**: Predicts the close price of Bitcoin and Ethereum using trained models.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/crypto_forecast.git
   cd crypto_forecast

# Create a virtual environment:

python -m venv crypto_forecast/include

# Activate the virtual environment:

crypto_forecast\include\Scripts\activate

# On Windows:

crypto_forecast\include\Scripts\activate

# On macOS/Linux:

source crypto_forecast/include/bin/activate

# Install dependencies:

pip install -r requirements.txt

# Usage

## Train Models:

* Use scripts in the src/models directory to train and save models

* Example: 
           python src/models/modeltraining.py


## Run Predictions:

* `Use inference.py to predict cryptocurrency close prices.`

* Example:
          
          python inference.py

# Requirements

* Python 3.8 or above

   * Libraries:
        pandas
        numpy
        scikit-learn
        xgboost

# Install all dependencies using:

  pip install -r requirements.txt


# Data

* Raw Data:
            
        * Located in data/raw_data/ directory.
        * Contains historical Bitcoin and Ethereum data.

* Processed Data:
 
        * Located in data/processed_data/ directory.
        * Cleaned and preprocessed data ready for model training.


# Models

       * Models are stored in the artifacts/ directory:
       * Linear Regression and XGBoost models for Bitcoin and Ethereum.