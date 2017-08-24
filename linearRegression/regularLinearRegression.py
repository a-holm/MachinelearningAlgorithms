# -*- coding: utf-8 -*-
"""Linear regression for machine learning

This file demonstrate knowledge of linear regression. Both by using
conventional libraries and by doing everything from scratch.The idea of linear
regression is to take continuous data and find the best fit of it to a line.

LOOK AT THE BOTTOM WHERE I WILL GO THROUGH LINEAR REGRESSION FROM SCRATCH.

Example:

        $ python linearRegression.py

Todo:
    *
"""

import datetime
import math
import pickle
from matplotlib import style
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quandl
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


style.use('ggplot')

# #############Setup data

# Import example data from quandl (stock prices in this case)
quandl.ApiConfig.api_key = 'ypdtiZxUzbE-p5t4yXEz'
df = quandl.get_table('WIKI/PRICES')
df.index = df['date']
df = df[['adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume']]
# Percent  point change from high to low
df['HL_PCT'] = (df['adj_high'] - df['adj_low']) / df['adj_low'] * 100.0
# Percent  point change from open to close
df['PCT_Change'] = (df['adj_close'] - df['adj_open']) / df['adj_open'] * 100.0
# Just the parts we want
df = df[['adj_close', 'HL_PCT', 'PCT_Change', 'adj_volume']]

forecast_col = 'adj_close'
# Remove nan and replace it with outliers.
df.fillna(-99999, inplace=True)
# To predict  (forecast out) price based on the previous 0.5%.
# Today (24.08.2017) this means  previous 100 when I import the quandl data.
forecast_out = int(math.ceil(0.005 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)  # forecast price column.

# ########### Training and test
x = np.array(df.drop(['label'], 1))  # features
x = preprocessing.scale(x)
x_lately = x[-forecast_out:]
x = x[:-forecast_out]
print(len(x_lately), len(x))


df.dropna(inplace=True)
y = np.array(df['label'])  # labels

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# clf = LinearRegression(n_jobs=-1)
# clf = svm.SVR(kernel='poly')
# clf.fit(x_train, y_train)
# saving the classifier,  which is smart to do on large datasets to avoid
# retraining algorithm. Not necessary now, but just showing for the portfolio.
# with open('linearregression.pickle', 'wb') as f:
#     pickle.dump(clf, f)
# to open:
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(x_test, y_test)
# print(accuracy)

# ########### Forecasting and predicting

forecast_set = clf.predict(x_lately)
# print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['adj_close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
