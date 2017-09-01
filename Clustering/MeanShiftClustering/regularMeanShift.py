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

Example:

        $ python regularMeanShift.py

Todo:
    *
"""
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("ggplot")

# Create training data
centers = [[1, 1, 1], [5, 5, 5], [3, 10, 10]]
X, _ = make_blobs(n_samples=100, centers=centers, cluster_std=1)

# Create Mean Shift object and fit data
ms = MeanShift()
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
print(cluster_centers)
n_clusters_ = len(np.unique(labels))
print("Number of estimated clusters: ", n_clusters_)

# Plot
colors = 10 * ['r', 'g', 'b', 'c', 'y', 'm']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')
ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
           marker="x", color='k', s=150, linewidths=5, zorder=10)
plt.show()
