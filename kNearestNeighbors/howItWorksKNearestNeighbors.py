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
# import matplotlib.pyplot as plt
from matplotlib import style
# from math import sqrt
import warnings
style.use('fivethirtyeight')

# hardcoded testdata
dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
new_features = [5, 7]

# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1], s=100)
# plt.show()


def k_nearest_neighbors(data, predict, k=3):
    """Function to calculate k nearest neighbors.

    Based on the parameter 'predict' we find the points in the local proximity
    of the training data and their label. In a larger dataset it would make
    sense to specify a radius to avoid going over all data points each time,
    but with the current dataset it does not matter so I avoid it to simplify.

    Args:
        data (dictionary): a dictionary where the keys are labels and the
                values are a list of lists of features.
        predict (list): a list of features that we will classify
        k (int): an int that is the amount of neighbors to be counted. Should
                be an odd number and higher than len(data) to avoid errors.

    Returns:
        str: The return value. The label that the predicted parameter has.


    """
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups')
    distances = []
    for group in data:
        for features in data[group]:
            # euclidean_distance = np.sqrt(np.sum((np.array(features)-np.array(predict))**2))
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))  # faster
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

result = k_nearest_neighbors(dataset, new_features, k=5)
print(result)
