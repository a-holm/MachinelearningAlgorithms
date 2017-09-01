# -*- coding: utf-8 -*-
"""Mean Shift unsupervised hierarchical classification for machine learning.

Mean Shift is very similar to the K-Means algorithm, except for one very
important factor: you do not need to specify the number of groups prior to
training. The Mean Shift algorithm finds clusters on its own. For this reason,
it is even more of an "unsupervised" machine learning algorithm than K-Means.

The way Mean Shift works is to go through each featureset
(a datapoint on a graph), and proceed to do a hill climb operation. Hill
Climbing is just as it sounds: The idea is to continually increase, or go up,
until you cannot anymore. We don't have for sure just one local maximal value.
We might have only one, or we might have ten. Our "hill" in this case will be
the number of featuresets/datapoints within a given radius. The radius is also
called a bandwidth, and the entire window is your Kernel. The more data within
the window, the better. Once we can no longer take another step without
decreasing the number of featuresets/datapoints within the radius, we take the
mean of all data in that region and we have located a cluster center. We do
this starting from each data point. Many data points will lead to the same
cluster center, which should be expected, but it is also possible that other
data points will take you to a completely separate cluster center.

This algorithm is usually used for research and finding structure and is not
expected to be super precise.

This file uses an imported titanic.xls file which contains non-numeric data and
shows how I would deal with such data. The data is found on the internet, but
the original source is unknown.

found it at the address:
http://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls

Example:

        $ python titanicMeanShift.py

Todo:
    *
"""
# import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing
import pandas as pd
style.use('ggplot')

"""
The data in the titanic.xls file looks like this:
------------------------------------------------
pclass - Passenger Class (1=1st; 2=2nd; 3=3rd)
survival - Survival (0=No; 1=Yes)
name - Name
sex - Sex
age - Age
sibsp - Number of Siblings/Spouses Aboard
parch - Number of Parents/Children Aboard
ticket - Ticket Number
fare - Passenger Fare (British pound)
cabin - Cabin
embarked - Port of Embarkation (C=Cherbourg; Q=Queenstown; S=Southampton)
boat - Lifeboat
body - Body Identification Number
home.dest - Home/Destination
"""

df = pd.read_excel('titanic.xls')
# print(df.head())
original_df = pd.DataFrame.copy(df)
df.drop(['body', 'name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
# print(df.head())


def handle_non_numerical_data(df):
    """Function to handle non-numerical data in the dataset."""
    columns = df.columns.values

    for column in columns:
        text_digit_values = {}

        def convert_to_int(val):
            """Convert non-numerical value to int."""
            return text_digit_values[val]
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            # Finding just the unique elements in current column
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_values:
                    # Create dict that contains unique number per unique string
                    text_digit_values[unique] = x
                    x += 1
            # map and replace all text strings with unique number
            df[column] = list(map(convert_to_int, df[column]))
    return df

df = handle_non_numerical_data(df)
# print(df.head())

# add/remove features to see what impact they have
df.drop(['boat'], 1, inplace=True)

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)
# use fitted model to label every row in original data
labels = clf.labels_
cluster_centers = clf.cluster_centers_
original_df['cluster_group'] = np.nan
for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]

# check accuracy
n_clusters = len(np.unique(labels))
survival_rates = {}
for i in range(n_clusters):
    temp_df = original_df[(original_df['cluster_group'] == float(i))]
    survival_cluster = temp_df[(temp_df['survived'] == 1)]
    survival_rate = len(survival_cluster) / len(temp_df)
    survival_rates[i] = (survival_rate, len(temp_df))
print(survival_rates)
