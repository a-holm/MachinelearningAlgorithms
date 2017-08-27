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

        $ python howItWorksMeanShift.py

Todo:
    *
"""
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')
colors = 10 * ["g", "r", "c", "b", "m", "y"]

X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8],
              [1, 0.6], [9, 11], [8, 2], [10, 2], [9, 3]])


class MeanShift:
    """Mean Shift class.

    This class is for creating an instance of Mean Shift. To avoid retraining
    or refitting (as it's also called) every time it is used.
    """

    def __init__(self, bandwidth=4):
        """The __init__ method of the Mean Shift class.

        Could have added a tolerance or max iteration, but don't think it's
        necessary and would just add more code. I have added tolerance and
        max iteration in other "from scratch" algorithms like the algorithm
        for K-means if you are interested in seing how I do that.

        Args:
            bandwidth (int): The radius, to see how it works in this context
                just read the doctype for the file.
        """
        self.bandwidth = bandwidth

    def fit(self, data):
        """Method to fit/train data and find clusters."""
        # find initial centroids which is each data point's location.
        centroids = {}
        for i in range(len(data)):
            centroids[i] = data[i]

        # optimize centroids
        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                for featureset in data:
                    # check how far each datapoint is from current centroid.
                    if np.linalg.norm(featureset - centroid) < self.bandwith:
                        in_bandwidth.append(featureset)
                new_centroid = np.average(in_bandwidth, axis=0)
                # make into tuple so I later can find unique arrays with set().
                new_centroids.append(tuple(new_centroid))
            uniques = sorted(set(new_centroids))
            # whittle down the centroids
            prev_centroids = dict(centroids)
            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])
            # check if optimized (check to see if the centroids have moved)
            optimized = True
            for i in centroids:
                # because the uniques are sorted we can do this:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                    break
            if optimized:
                break
        self.centroids = centroids

        def predict(self, data):
            pass

# Create MeanShift instance and fit data
clf = MeanShift()
clf.fit(X)
centroids = clf.centroids
# Plot data
plt.scatter(X[:, 0], X[:, 1], s=150)
for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='x', s=150)
plt.show()
