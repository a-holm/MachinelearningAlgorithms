# -*- coding: utf-8 -*-
"""Simple linear regression for machine learning.

This file demonstrate knowledge of linear regression. By using
conventional libraries.The idea of linear regression is to take continuous
data and find the best fit of it to a line.

Simple linear regression just refers to the fact that the features only
includes one column. So the label is composed by just one variable and one
constant.

Example:

        $ python regularLinearRegression2.py

Todo:
    *
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, 1].values

# Splitting the Dataset into a Training set and a Test set
feature_train, feature_test, label_train, label_test = train_test_split(
    features, labels, test_size=1 / 3.0)

# Fitting the Training set with Simple Linear regression model.
regressor = LinearRegression()
regressor.fit(feature_train, label_train)

# Predicting the Test set results
label_pred = regressor.predict(feature_test)

# Visualising the regression line and training set
plt.scatter(feature_train, label_train, color='y')
plt.plot(feature_train, regressor.predict(feature_train), color='r')
plt.title('Experience vs Salary (training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the regression line and test set
plt.scatter(feature_test, label_test, color='c')
plt.plot(feature_test, regressor.predict(feature_test), color='b')
plt.title('Experience vs Salary (testing set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
