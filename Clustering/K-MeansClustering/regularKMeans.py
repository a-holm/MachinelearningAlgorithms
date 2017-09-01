# -*- coding: utf-8 -*-
"""K-Means unsupervised classification for machine learning.

K-means clustering is a unsupervised method to cluser or group the data.
K-means allows you to choose the number (k) of categories/groups and
categorizes it automatically when it has come up with solid categories.

This algorithm is usually used for research and finding structure and is not
expected to be super precise.

Example:

        $ python regularKMeans.py

Todo:
    *
"""
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans

style.use('ggplot')
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

clf = KMeans(n_clusters=5)
clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_

colors = ["g.", "r.", "c.", "b.", "k."]

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, linewidths=5)
plt.show()
