# -*- coding: utf-8 -*-
"""K-Means unsupervised classification for machine learning.

K-means clustering is a unsupervised method to cluser or group the data.
K-means allows you to choose the number (k) of categories/groups and
categorizes it automatically when it has come up with solid categories.

This algorithm is usually used for research and finding structure and is not
expected to be super precise.

Example:

        $ python howItWorksKMeans.py

Todo:
    *
"""
# import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')
colors = ["g.", "r.", "c.", "b.", "k."]

X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])


class KMeans:
    """K-means class.

    This class is for creating an instance of K Means. To avoid retraining or
    refitting (as it's also called) every time it is used.
    """

    def __init__(self, k=2, tol=0.001, max_iter=300):
        """The __init__ method of the K-means class.

        Args:
            k (int): The number of clusters/groups that we are looking for
            tol (float): tolerance is basically how much the centroid is going
                to move.
            max_iter (int): Max iteration is basically how many times we run
                this algorithm before we decide it's enough.
        """
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        """Method to fit/train data and find clusters."""
        self.centroids = {}

        for i in range(self.k):
            # Centroids are two first in data, if random is wanted then
            # shuffle data randomly before fitting.
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}
            for i in range(self.k):
                self.classifications[i] = []
            for featureset in data:
                distances = [
                    np.linalg.norm(featureset - self.centroids[centroid])
                    for centroid in self.centroids]
                print(distances)
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
            prev_centroids = dict(self.centroids)
            for classification in self.classifications:
                self.centroids[classification] = np.average(
                    self.classifications[classification], axis=0)

            optimized = True
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) /
                          original_centroid * 100) > self.tol:
                    optimized = False
            if optimized:
                break

    def predict(self, data):
        """Method to predict what cluster data belongs to."""
        pass
