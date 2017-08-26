# -*- coding: utf-8 -*-
"""Soft Margin SVM classification with kernels for machine learning.

Soft margin SVM is basically an SVM (see folder **supportVectorMachine**) which
has some 'slack' and allows features to be 'wrongly' classified to avoid
overfitting the classifier. This also includes kernels. Kernels use the inner
product to help us transform the feature space to make it possible for Support
Vector Machines to create a good hyperplane with non-linear feature sets.

This file uses sklearn and other common python libraries to solve the SVM and
includes contemporary ways to use SVMs, including how to separate more than two
classes of data with One-vs-rest (OVR) and One-vs-One (OVO) because SVMs are
binary classifiers so intially they only classify into two classes.

dataset is breast cancer data from: http://archive.ics.uci.edu/ml/datasets.html

Example:

        $ python regularSoftMarginSVM.py.py

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

clf = svm.SVC(kernel='linear', C=100.0)  # Linear kernel with soft margin
clf.fit(X_train, y_train)

# Could have saved in a pickle, but not a very large data set.

accuracy = clf.score(X_test, y_test)
print(accuracy)
