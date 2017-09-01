# -*- coding: utf-8 -*-
"""K Nearest Neighbors classification for machine learning.

This file demonstrate knowledge of K Nearest Neighbors classification. By
building the algorithm from scratch.
The idea of K Nearest Neighbors classification is to best divide and separate
the data based on clustering the data and classifying based on the proximity
to it's K closest neighbors and their classifications.

'Closeness' is measured by the euclidean distance.

dataset is breast cancer data from: http://archive.ics.uci.edu/ml/datasets.html

Example:

        $ python howItWorksKNearestNeighbors.py

Todo:
    *
"""
from collections import Counter
import numpy as np
import pandas as pd
import random
import warnings


# hardcoded testdata
dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
new_features = [5, 7]


def k_nearest_neighbors(data, predict, k=3):
    """Function to calculate k nearest neighbors.

    Based on the parameter 'predict' we find the points in the local proximity
    of the training data and their label. In a larger dataset it would make
    sense to specify a radius to avoid going over all data points each time,
    but with the current dataset it does not matter so I avoid it to simplify.

    Can work on both linear and non-linear data sets.

    Args:
        data (dictionary): a dictionary where the keys are labels and the
                values are a list of lists of features.
        predict (list): a list of features that we will classify
        k (int): an int that is the amount of neighbors to be counted. Should
                be an odd number and higher than len(data) to avoid errors.

    Returns:
        str: The return value. The label that the predicted parameter has.


    """
    predict = np.array(predict)
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups')
    distances = []
    for group in data:
        for features in data[group]:
            features = np.array(features)
            # euclidean_distance = np.sqrt(np.sum(features-predict)**2))
            euclidean_distance = np.linalg.norm(features - predict)  # faster
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k

    return vote_result, confidence

# Setup data
df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

# make test and training data
test_size = 0.2
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
train_data = full_data[:-int(test_size * len(full_data))]  # first 80%
test_data = full_data[-int(test_size * len(full_data)):]  # last 20%

# populate test_set and train_set
for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])


# test Accuracy
correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote, conf = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        else:
            print(conf)
        total += 1
print('Accuracy: ', correct / total)
