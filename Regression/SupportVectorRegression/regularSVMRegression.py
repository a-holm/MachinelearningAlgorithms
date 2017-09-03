# -*- coding: utf-8 -*-
"""Support Vector regression for machine learning.

Support Vector Machine can also be used as a regression method, maintaining all
the main features that characterize the algorithm (maximal margin). The Support
Vector Regression (SVR) uses the same principles as the SVM for classification,
with only a few minor differences. First of all, because output is a real
number it becomes very difficult to predict the information at hand, which has
infinite possibilities. In the case of regression, a margin of tolerance is set
in approximation to the SVM which would have already requested from the
problem. But besides this fact, there is also a more complicated reason, the
algorithm is more complicated therefore to be taken in consideration. However,
the main idea is always the same: to minimize error, individualizing the
hyperplane which maximizes the margin, keeping in mind that part of the error
is tolerated.

Example:

        $ python regularSVMRegression.py

Todo:
    *
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split


# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
features = dataset.iloc[:, 1:2].values
labels = dataset.iloc[:, 2].values

# Splitting the Dataset into a Training set and a Test set
"""feature_train, feature_test, label_train, label_test = train_test_split(
    features, labels, test_size=0.2)"""

# Feature scaling, normalize scale is important. Especially on algorithms
# involving euclidian distance. Two main feature scaling formulas are:
# Standardisation: x_stand = (x-mean(x))/(standard_deviation(x))
# Normalisation: x_norm = (x-min(x))/(max(x)-min(x))
sc_features = StandardScaler()
sc_labels = StandardScaler()
features = sc_features.fit_transform(features)
labels = sc_labels.fit_transform(labels.reshape(-1, 1))

# Fit the SVR regression model to the dataset
regressor = SVR(kernel='rbf')
regressor.fit(features, labels)

# Predict new result with the SVR regression model
# y_pred = sc_labels.inverse_transform(regressor.predict(
#     sc_features.transform(np.array([65]).reshape(-1, 1))))
x_pred = sc_features.transform(np.array([6.5]).reshape(-1, 1))
y_pred = regressor.predict(x_pred)

# Visualising the regression results with smoother curve
x_grid = np.arange(min(features), max(features), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(features, labels, color='r')
plt.scatter(x_pred, y_pred, color='c')
plt.plot(x_grid, regressor.predict(x_grid), color='b')
plt.title('Truth or Bluff (SVR regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
