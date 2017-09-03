# -*- coding: utf-8 -*-
"""Data Preprocessing Template.

This is a file that I use as a template for data pre-processing in Machine
learning projects. After a while I figured out it might be easier to have it as
a reference. And to save time.

Example:

        $ python dataPreprocessingTemplate.py

Todo:
    *
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import (Imputer,
                                   LabelEncoder,
                                   OneHotEncoder,
                                   StandardScaler)
from sklearn.model_selection import train_test_split

# importing the dataset
dataset = pd.read_csv('data/Data.csv')
features = dataset.iloc[:, :-1].values  # Country, Age, Salary
labels = dataset.iloc[:, -1].values  # Purchased

# Splitting the Dataset into a Training set and a Test set
feature_train, feature_test, label_train, label_test = train_test_split(
    features, labels, test_size=0.2)

# Feature scaling, normalize scale is important. Especially on algorithms
# involving euclidian distance. Two main feature scaling formulas are:
# Standardisation: x_stand = (x-mean(x))/(standard_deviation(x))
# Normalisation: x_norm = (x-min(x))/(max(x)-min(x))
"""sc_feature = StandardScaler()
feature_train = sc_feature.fit_transform(feature_train)
feature_test = sc_feature.transform(feature_test)"""

"""# Dealing with missing data:
# In other projects I have just made them into outliers and filtered them out.
# Another technique is to give the numbers the value of the mean of the column.
strategy = 'mean'  # can be 'mean'(default), 'median' or 'most frequent'
# calculating the strategy on column if axis = 0, on row if axis = 1
imputer = Imputer(missing_values='NaN', strategy=strategy, axis=0)
imputer = imputer.fit(features[:, 1:3])
features[:, 1:3] = imputer.transform(features[:, 1:3])"""


"""# Encode categorical data and make them into numbers
labelEncoder_x = LabelEncoder()
# enough if there are hierarchical relations between rows
features[:, 0] = labelEncoder_x.fit_transform(features[:, 0])
# If vales are categorical then make dummy columns to avoid attributing order
onehotencoder = OneHotEncoder(categorical_features=[0])
features = onehotencoder.fit_transform(features).toarray()
labelEncoder_y = LabelEncoder()
labels = labelEncoder_y.fit_transform(labels)  # no need for more because label
print(labels)
"""
