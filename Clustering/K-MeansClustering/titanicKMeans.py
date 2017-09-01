# -*- coding: utf-8 -*-
"""K-Means unsupervised classification for machine learning.

K-means clustering is a unsupervised method to flatly cluser or group the data.
K-means allows you to choose the number (k) of categories/groups and
categorizes it automatically when it has come up with solid categories.

This algorithm is usually used for research and finding structure and is not
expected to be super precise.

This file uses an imported titanic.xls file which contains non-numeric data and
shows how I would deal with such data. The data is found on the internet, but
the original source is unknown.

found it at the address:
http://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls

Example:

        $ python titanicKMeans.py

Todo:
    *
"""
# import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
style.use('ggplot')

"""
The data in the titanic.xls file looks like this:
------------------------------------------------
pclass - Passenger Class (1=1st; 2=2nd; 3=3rd)
survival - Survival (0=No; 1=Yes)
name - Name
sex - Sex
age - Age
sibsp - Number of Siblings/Spouses Aboard
parch - Number of Parents/Children Aboard
ticket - Ticket Number
fare - Passenger Fare (British pound)
cabin - Cabin
embarked - Port of Embarkation (C=Cherbourg; Q=Queenstown; S=Southampton)
boat - Lifeboat
body - Body Identification Number
home.dest - Home/Destination
"""

df = pd.read_excel('titanic.xls')
# print(df.head())
df.drop(['body', 'name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
# print(df.head())


def handle_non_numerical_data(df):
    """Function to handle non-numerical data in the dataset."""
    columns = df.columns.values

    for column in columns:
        text_digit_values = {}

        def convert_to_int(val):
            """Convert non-numerical value to int."""
            return text_digit_values[val]
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_values:
                    text_digit_values[unique] = x
                    x += 1
            df[column] = list(map(convert_to_int, df[column]))
    return df

df = handle_non_numerical_data(df)
# print(df.head())
df.drop(['boat'], 1, inplace=True)  # drop Lifeboat col because it has effect.

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct / len(X))
