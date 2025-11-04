from flask import Flask, render_template
import pickle
import pandas as pd
import os
import requests
import yfinance as yf
import io
from datetime import date, timedelta

app = Flask(__name__)

# Load model once
try:
    model = pickle.load(open("trained_model.sav", "rb"))
except FileNotFoundError:
    model = None
    print("⚠️ trained_model.sav not found")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/pridict')
def pridict():
    SYMBOL = "GOOGL"
    yesterday = date.today() - timedelta(days=1)
    start_date = yesterday - timedelta(days=11)

    # Try using Alpha Vantage if API key is available (Render)
    API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")
    data = None

    if API_KEY:
        print("📡 Using Alpha Vantage API")
        try:
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={SYMBOL}&outputsize=compact&datatype=csv&apikey={API_KEY}'
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = pd.read_csv(io.StringIO(response.text))
            data = data.rename(columns={
                'timestamp': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'adjusted_close': 'Close',
                'volume': 'Volume'
            })
            data = data[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]
            data = data.sort_values(by='Date', ascending=True).reset_index(drop=True)
        except Exception as e:
            print("Alpha Vantage fetch failed:", e)

    # Fallback to yfinance (local environment)
    if data is None or data.empty:
        print("📊 Using yfinance fallback")
        try:
            data = yf.download(SYMBOL, start=start_date, end=yesterday, progress=False, threads=False)
            data.reset_index(inplace=True)
            data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        except Exception as e:
            return render_template('error.html', message=f"Data fetch failed: {e}")

    if data.empty:
        return render_template('error.html', message="Stock data unavailable. Try again later.")

    # --- Feature Engineering ---
    data['Price Changed'] = data['Close'] - data["Open"]
    data['MA7'] = data['Close'].rolling(7).mean()
    past = data['Close'].mean()
    g = data.copy()
    g['Close'] = g['Close'].round()
    x = g['Date'].astype(str).tolist()
    Y = g['Close'].tolist()

    data = data.dropna()
    data = data.drop(['Date', 'Close'], axis=1)
    data = data.tail(1)

    y_pred = model.predict(data)
    y_pred_val = round(float(y_pred[0]), 2)

    if y_pred_val > past:
        signal = "Bullish momentum — consider buying."
    elif y_pred_val < past:
        signal = "Bearish momentum — consider selling."
    else:
        signal = "Stable movement — consider holding."

    return render_template(
        'pridict.html',
        signal=signal,
        x=x,
        Y=Y,
        y_pred=f"{y_pred_val}$"
    )


if __name__ == '__main__':
    app.run(debug=True)
