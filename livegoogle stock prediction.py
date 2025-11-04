# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
from datetime import date, timedelta

#loading the model
loaded_model=pickle.load(open("C:/Users/Barathkumar/Downloads/trained_model.sav",'rb'))

yesterday = date.today() - timedelta(days=1)
start_date = yesterday - timedelta(days=9)

data = yf.download("GOOGL", start=start_date, end=yesterday)

data.reset_index(inplace=True)
data=pd.DataFrame(data)

data.columns.names = [None, None]

data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
data['Price Changed']=data['Close']-data["Open"]
data['MA7'] = data['Close'].rolling(7).mean()

data=data.dropna()
data=data.drop('Date',axis=1)


data=data.drop('Close',axis=1)
y_pred = loaded_model.predict(data)
print(y_pred) 