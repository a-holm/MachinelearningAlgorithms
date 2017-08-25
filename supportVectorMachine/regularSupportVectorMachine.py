# -*- coding: utf-8 -*-
"""Support Vector Machine (SVM) classification for machine learning.

SVM is a binary classifier. The objective of the SVM is to find the best
separating hyperplane in vector space which is also referred to as the
decision boundary. And it decides what separating hyperplane is the 'best'
because the distance from it and the associating data it is separating is the
greatest at the plane in question.

This is the file where I create use scikit-learn to use the algorithm.

dataset is breast cancer data from: http://archive.ics.uci.edu/ml/datasets.html

Example:

        $ python regularSupportVectorMachine.py

Todo:
    *
"""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)  # make missing attribute values outliers
df.drop(['id'], 1, inplace=True)  # remove useless column

X = np.array(df.drop(['class'], 1))  # features
y = np.array(df['class'])  # labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)

# Could have saved in a pickle, but not a very large data set.

accuracy = clf.score(X_test, y_test)
print(accuracy)

example1 = [4, 2, 1, 1, 1, 2, 3, 2, 1]
example2 = [4, 2, 1, 2, 2, 2, 3, 2, 1]

example_measures = np.array([example1, example2])
example_measures = example_measures.reshape(len(example_measures), -1)

prediction = clf.predict(example_measures)
print(prediction)
