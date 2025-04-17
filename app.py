import os
from flask import Flask, render_template, request
import requests
import pandas as pd
import pickle
from datetime import datetime

app = Flask(__name__)

def fetch_historical_data(symbol, selected_date):
    url = "https://api.binance.com/api/v3/klines"
    start_time = int(selected_date.timestamp() * 1000)
    end_time = start_time + 86400 * 1000  # Add 24 hours in milliseconds
    params = {
        'symbol': symbol,
        'interval': '1d',
        'startTime': start_time,
        'endTime': end_time,
        'limit': 1
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data[0] if data else None
    except requests.exceptions.RequestException as e:
        return None

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

def load_artifacts(symbol, model_type):
    prefix = symbol[:3].lower()
    try:
        with open(os.path.join('artifacts', f'{prefix}_scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join('artifacts', f'{prefix}_{model_type}.pkl'), 'rb') as f:
            model = pickle.load(f)
        return scaler, model
    except Exception as e:
        raise Exception(f"Error loading artifacts: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if not all(key in request.form for key in ['date', 'dataset', 'model']):
            return render_template('index.html', error="Missing form data")

        date_str = request.form.get('date', '')
        symbol = request.form.get('dataset', 'BTCUSDT')
        model_type = request.form.get('model', 'linear')

        try:
            selected_date = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return render_template('index.html', error="Invalid date format")
        
        if symbol not in ['BTCUSDT', 'ETHUSDT']:
            return render_template('index.html', error="Invalid cryptocurrency selection")

        if model_type not in ['linear', 'xgboost']:
            return render_template('index.html', error="Invalid model selection")

        kline = fetch_historical_data(symbol, selected_date)
        if not kline:
            return render_template('index.html', error="No data available for selected date")
        
        try:
            features = prepare_features(kline)
        except Exception as e:
            return render_template('index.html', error=f"Feature preparation failed: {str(e)}")
        
        try:
            scaler, model = load_artifacts(symbol, model_type)
            scaled_features = scaler.transform(features.values)
            prediction = model.predict(scaled_features)[0]
            prediction = round(prediction, 2)
        except Exception as e:
            return render_template('index.html', error=f"Prediction failed: {str(e)}")
        
        next_day = selected_date.replace(hour=0, minute=0, second=0) + pd.DateOffset(days=1)
        return render_template('index.html', 
                            prediction=prediction,
                            symbol=symbol[:3],
                            model=model_type,
                            selected_date=selected_date.strftime('%Y-%m-%d'),
                            next_day=next_day.strftime('%Y-%m-%d'))
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)