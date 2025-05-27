from flask import Flask, render_template, request
import requests
import pandas as pd
import pickle
import os
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

def load_artifacts(prefix):
    artifacts_dir = os.path.join(BASE_DIR, 'artifacts')
    scaler_path = os.path.join(artifacts_dir, f'{prefix}_scaler.pkl')
    xgb_path    = os.path.join(artifacts_dir, f'{prefix}_xgboost.pkl')
    lr_path     = os.path.join(artifacts_dir, f'{prefix}_linear.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(xgb_path, 'rb') as f:
        xgb = pickle.load(f)
    with open(lr_path, 'rb') as f:
        lr = pickle.load(f)
    return scaler, xgb, lr


ARTIFACTS = {
    'BTC': load_artifacts('btc'),
    'ETH': load_artifacts('eth')
}

def fetch_kline_on_date(symbol, date_str):
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    start_ms = int(dt.timestamp() * 1000)
    end_ms   = start_ms + 24 * 60 * 60 * 1000
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': '1d',
        'startTime': start_ms,
        'endTime': end_ms,
        'limit': 1
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise ValueError(f'No data for {symbol} on {date_str}')
    return data[0]

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

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        date = request.form['date']  
        dataset = request.form['dataset'] 
        model = request.form['model']
        symbol = f"{dataset}USDT"
        scaler, xgb, lr = ARTIFACTS[dataset]
        kline = fetch_kline_on_date(symbol, date)
        features = prepare_features(kline)
        scaled = scaler.transform(features.values)
        if model == 'xgboost':
            pred = float(xgb.predict(scaled)[0])
        else:
            pred = float(lr.predict(scaled)[0])
        result = {
            'date': date,
            'dataset': dataset,
            'model': model,
            'prediction': round(pred, 2)
        }
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')