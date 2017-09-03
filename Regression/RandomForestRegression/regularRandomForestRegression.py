# -*- coding: utf-8 -*-
"""Random Forest Regression for machine learning.

Random forest algorithm is a supervised classification algorithm. As the name
suggest, this algorithm creates the forest with a number of decision trees.

In general, the more trees in the forest the more robust the forest looks like.
In the same way in the random forest classifier, the higher the number of trees
in the forest gives the high accuracy results.


Example:

        $ python regularRandomForestRegression.py

Todo:
    *
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split


# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
features = dataset.iloc[:, 1:2].values
labels = dataset.iloc[:, 2].values

# Splitting the Dataset into a Training set and a Test set
"""feature_train, feature_test, label_train, label_test = train_test_split(
    features, labels, test_size=0.2)
"""
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

# Fit the Random Forest Regression to the dataset
regressor = RandomForestRegressor(n_estimators=310, random_state=0)
regressor.fit(features, labels)


# Predict new result with the Random Forest Regression
y_pred = regressor.predict(6.5)


# Visualising the regression results with smoother curve
x_grid = np.arange(min(features), max(features), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(features, labels, color='r')
plt.plot(x_grid, regressor.predict(x_grid), color='b')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
