# -*- coding: utf-8 -*-
"""Logistic Regression Classification for machine learning.

In statistics, logistic regression, or logit regression, or logit model is a
regression model where the dependent variable (DV) is categorical. This project
covers the case of a binary dependent variable—that is, where it can take
only two values, "0" and "1", which represent outcomes such as pass/fail,
win/lose, alive/dead or healthy/sick.

The binary Logistic regression model is an example of a qualitative
response/discrete choice model. It is used to estimate the probability of a
binary response based on one or more predictor (or independent) variables
(features).

Example:

        $ python regularLogisticRegression.py

Todo:
    *
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
features = dataset.iloc[:, [2, 3]].values  # Country, Age, Salary
labels = dataset.iloc[:, 4].values  # Purchased

# Splitting the Dataset into a Training set and a Test set
feature_train, feature_test, label_train, label_test = train_test_split(
    features, labels, test_size=0.25)

# Feature scaling, normalize scale is important. Especially on algorithms
# involving euclidian distance. Two main feature scaling formulas are:
# Standardisation: x_stand = (x-mean(x))/(standard_deviation(x))
# Normalisation: x_norm = (x-min(x))/(max(x)-min(x))
sc_feature = StandardScaler()
feature_train = sc_feature.fit_transform(feature_train)
feature_test = sc_feature.transform(feature_test)

# Fitting the Logistic Regression to the dataset
classifier = LogisticRegression()
classifier.fit(feature_train, label_train)

# Predicting the results of the Test set
y_pred = classifier.predict(feature_test)

# Creating the Confusion Matrix
cm = confusion_matrix(label_test, y_pred)

# Visualize the Training set results
"""X_set, y_set = feature_train, label_train
X1, X2 = np.meshgrid(
    np.arange(
        start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01
    ),
    np.arange(
        start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01
    )
)
plt.contourf(
    X1, X2, classifier.predict(
        np.array([X1.ravel(), X2.ravel()]).T
    ).reshape(X1.shape), alpha=0.75, cmap=ListedColormap(('red', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'blue'))(i), label=j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()"""


# Visualize the Test set results
X_set, y_set = feature_test, label_test
X1, X2 = np.meshgrid(
    np.arange(
        start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01
    ),
    np.arange(
        start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01
    )
)
plt.contourf(
    X1, X2, classifier.predict(
        np.array([X1.ravel(), X2.ravel()]).T
    ).reshape(X1.shape), alpha=0.75, cmap=ListedColormap(('red', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'blue'))(i), label=j)
plt.title('Logistic Regression (Testing set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
