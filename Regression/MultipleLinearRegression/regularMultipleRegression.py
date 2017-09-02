# -*- coding: utf-8 -*-
"""Multiple linear regression for machine learning.

A linear regression model that contains more than one predictor variable is
called a multiple linear regression model. It is basically the same as Simple
Linear regression, but with more predictor variables (features). The idea is
that linearly related predictor variables can approximate the labels with a
'best fitted' hyperplane or surface.

Example:

        $ python regularMultipleRegression.py

Todo:
    *
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import (LabelEncoder,
                                   OneHotEncoder)
from sklearn.model_selection import train_test_split

# importing the dataset
dataset = pd.read_csv('50_Startups.csv')
features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, 4].values

# encode State column
labelencoder_features = LabelEncoder()
features[:, 3] = labelencoder_features.fit_transform(features[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
features = onehotencoder.fit_transform(features).toarray()

# Avoiding the Dummy variable trap
features = features[:, 1:]  # is done automatically, but just to show I know

# Splitting the Dataset into a Training set and a Test set
feature_train, feature_test, label_train, label_test = train_test_split(
    features, labels, test_size=0.2)

# Fit the training set naively with the mutliple linear regression model
regressor = LinearRegression()
regressor.fit(feature_train, label_train)
# Predict the test set
label_pred = regressor.predict(feature_test)

# Building the optimal model using the Backward Elimenation method
# Due to statsmodels we need to add an intercept column
features = np.append(arr=np.ones((50, 1)).astype(int), values=features, axis=1)
columnlist = list(range(features.shape[1]))  # liste med num rader
significant = 0.05
while True:
    features_opt = features[:, columnlist]
    regressor_OLS = sm.OLS(endog=labels, exog=features_opt).fit()
    pvalues = regressor_OLS.pvalues
    if (np.max(pvalues) > significant):
        i = int(np.where(pvalues == np.max(pvalues))[0])
        columnlist.pop(i)
    else:
        break
regressor_OLS.summary()
