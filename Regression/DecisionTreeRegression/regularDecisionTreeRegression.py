# -*- coding: utf-8 -*-
"""Decision Tree regression for machine learning.

Decision tree builds regression or classification models in the form of a tree
structure. It brakes down a dataset into smaller and smaller subsets while at
the same time an associated decision tree is incrementally developed. The final
result is a tree with decision nodes and leaf nodes.

Example:

        $ python regularDecisionTreeRegression.py

Todo:
    *
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
# from sklearn.preprocessing import StandardScaler
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
"""sc_feature = StandardScaler()
feature_train = sc_feature.fit_transform(feature_train)
feature_test = sc_feature.transform(feature_test)
sc_labels = StandardScaler()
labels_train = sc_labels.fit_transform(labels_train)
labels_test = sc_labels.transform(labels_test)
"""

# Fit the Decision Tree Regression to the dataset
regressor = DecisionTreeRegressor(random_state=0)  # random_state for debug
regressor.fit(features, labels)

# Predict new result with the Decision Tree Regression
y_pred = regressor.predict(6.5)

# Visualising the regression results with higher resolution
x_grid = np.arange(min(features), max(features), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(features, labels, color='r')
plt.plot(x_grid, regressor.predict(x_grid), color='b')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
