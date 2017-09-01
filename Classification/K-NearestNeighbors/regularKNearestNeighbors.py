# -*- coding: utf-8 -*-
"""K Nearest Neighbors classification for machine learning.

This file demonstrate knowledge of K Nearest Neighbors classification. By using
the standard Python methods for the algorithm.
The idea of K Nearest Neighbors classification is to best divide and separate
the data based on clustering the data and classifying based on the proximity
to it's K closest neighbors and their classifications.

dataset is breast cancer data from: http://archive.ics.uci.edu/ml/datasets.html

Example:

        $ python regularKNearestNeighbors.py

Todo:
    *
"""
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)  # make missing attribute values outliers
df.drop(['id'], 1, inplace=True)  # remove useless column

X = np.array(df.drop(['class'], 1))  # features
y = np.array(df['class'])  # labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

# Could have saved in a pickle, but not a very large data set.

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([4, 2, 1, 1, 1, 2, 3, 2, 1])
example_measures = example_measures.reshape(len(example_measures), -1)

prediction = clf.predict(example_measures)
print(prediction)
