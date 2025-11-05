from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import requests
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
REQUIRED_FEATURES = ['High', 'Low', 'Open', 'Volume', 'Price Changed', 'MA7']
@app.route('/pridict')
def pridict():
    # --- Download last 10 days of stock data ---
    EODHD_API_KEY = "690a34e60b9c65.54263714"
    SYMBOL = "GOOGL.US"
    today = date.today()
    start_date = today - timedelta(days=20)  # Fetch more than 11 days to handle weekends/holidays
    end_date = today - timedelta(days=1)

    EODHD_URL = (
        f"https://eodhd.com/api/eod/{SYMBOL}?"
        f"from={start_date}&to={end_date}&"
        f"api_token={EODHD_API_KEY}&fmt=json"
    )

    try:
        r = requests.get(EODHD_URL)
        r.raise_for_status()
        raw_data = r.json()

        # Convert EODHD JSON response to DataFrame
        data_pred = pd.DataFrame(raw_data)

        if data_pred.empty:
            print("EODHD data fetch returned an empty list.")
            raise ValueError("EODHD API returned no data.")

        # Standardize column names (EODHD uses lowercase names)
        data_pred.rename(columns={
            'date': 'Date', 'close': 'Close', 'high': 'High',
            'low': 'Low', 'open': 'Open', 'volume': 'Volume'
        }, inplace=True)

        # Ensure numeric types
        for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
            data_pred[col] = pd.to_numeric(data_pred[col], errors='coerce')

        # Sort data chronologically (EODHD typically returns newest last)
        data_pred = data_pred.sort_values(by='Date').reset_index(drop=True)

        # --- Feature Engineering (Must match training) ---
        data_pred['Price Changed'] = data_pred['Close'] - data_pred["Open"]
        data_pred['MA7'] = data_pred['Close'].rolling(7).mean()

        # Drop NaNs created by rolling mean
        data_pred = data_pred.dropna().reset_index(drop=True)

        # Get the latest row for prediction
        latest_data_row = data_pred.tail(1)

        # Get historical average (all closing prices *before* the latest point)
        past = data_pred['Close'].iloc[:-1].mean()

        # 5. Prepare prediction features with REQUIRED_FEATURES order
        X_pred_new = latest_data_row[REQUIRED_FEATURES]

        # --- Prediction ---
        y_pred_new = model.predict(X_pred_new)
        y_pred_new = float(y_pred_new[0]
        # --- Signal and Output ---
        if y_pred_new > past:
            signal = "Bullish momentum — consider buying."
        elif y_pred_new < past:
            signal = "Bearish momentum — consider selling."
        else:
            signal = "Stable movement — consider holding."



        # --- Graphing (Adjusted to use the reliable EODHD data) ---
        # Plotting the last 7 *available* days
        g = data_pred.tail(7).copy()
        x_plot = g["Date"].astype(str).tolist()
        Y_plot = g['Close'].round(2).tolist()



    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from EODHD: {e}")
    except ValueError as e:
        print(f"Data processing error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
    # ✅ Convert to list before sending
    return render_template(
        'pridict.html',
        signal=signal,
        x=x_plot,
       Y=Y_plot,
        y_pred=y_pred_new
    )

if __name__ == '__main__':
    app.run(debug=True)
