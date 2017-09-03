# -*- coding: utf-8 -*-
"""Polynomial regression for machine learning.

polynomial regression is a form of regression analysis in which the
relationship between the independent variable x and the dependent variable y is
modelled as an nth degree polynomial in x. Polynomial regression fits a
nonlinear relationship between the value of x and the corresponding conditional
mean of y, denoted E(y |x)Although polynomial regression fits a nonlinear model
to the data, as a statistical estimation problem it is linear, in the sense
that the regression function E(y | x) is linear in the unknown parameters that
are estimated from the data. For this reason, polynomial regression is
considered to be a special case of multiple linear regression.

Example:

        $ python regularPolynomialRegression.py

Todo:
    *
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
features = dataset.iloc[:, 1:2].values
labels = dataset.iloc[:, 2].values

# No need to split the dataset into a Training set and a Test set
# No need for feature scaling in this example

# Fit Polynomial regression to the dataset
poly_reg = PolynomialFeatures(degree=5)
feature_poly = poly_reg.fit_transform(features)
lin_reg2 = LinearRegression()
lin_reg2.fit(feature_poly, labels)

# Visualising the Linear regression result
# plt.scatter(features, labels, color='r')
# plt.plot(features, lin_reg.predict(features), color='b')
# plt.title('Truth or Bluff (Linear Regression)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()

# Visualising the Polynomial regression result
# plt.scatter(features, labels, color='g')
# plt.plot(features, lin_reg2.predict(feature_poly), color='c')
# plt.title('Truth or Bluff (Polynomial Regression)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()

# Predict new result with Polynomial regression
lin_reg2.predict(poly_reg.fit_transform(6.5))
