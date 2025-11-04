from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import matplotlib.pyplot as mp
import yfinance as yf

from datetime import date, timedelta
import io, base64

app = Flask(__name__)

# Load the trained model once
model = pickle.load(open("trained_model.sav", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/pridict')
def pridict():
    # --- Download last 10 days of stock data ---
    yesterday = date.today() - timedelta(days=1)
    start_date = yesterday - timedelta(days=11)

    data = yf.download("GOOG", start=start_date, end=yesterday, progress=False)
    if data.empty:
        return render_template('error.html', message="Stock data not available yet. Please try again later.")
    data.reset_index(inplace=True)
    data = pd.DataFrame(data)

    data.columns.names = [None, None]

    data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

    data['Price Changed'] = data['Close'] - data["Open"]
    past = data['Close'].mean()
    data['MA7'] = data['Close'].rolling(7).mean()
    g = data
    g['Close'] = g['Close'].round()
    x = g["Date"]
    Y = g['Close']


    data = data.dropna()
    data = data.drop('Date', axis=1)

    print(past)
    y = data['Close']
    data = data.drop('Close', axis=1)
    data = data.tail(1)
    y_pred = model.predict(data)

    if y_pred > past:
        signal = "Bullish momentum — consider buying."
    elif y_pred < past:
        signal = "Bearish momentum — consider selling."
    else:
        signal = "Stable movement — consider holding."
    x=x.astype(str).tolist()
    Y=Y.tolist()
    y_pred = round(float(y_pred[0]), 2)  # round to 2 decimals
    y_pred = f"{y_pred}$"
    # ✅ Convert to list before sending
    return render_template(
        'pridict.html',
        signal=signal,
        x=x,
       Y=Y,
        y_pred=y_pred
    )

if __name__ == '__main__':
    app.run(debug=True)
