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
import random
from sklearn.datasets.samples_generator import make_blobs

style.use('ggplot')
colors = 10 * ["g", "r", "c", "b", "m", "y"]
# Create test data
# X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8],
#               [1, 0.6], [9, 11], [8, 2], [10, 2], [9, 3]])
centers = random.randrange(2, 5)
X, y = make_blobs(n_samples=50, centers=centers, n_features=2)
print(centers)


class MeanShift:
    """Mean Shift class.

    This class is for creating an instance of Mean Shift. To avoid retraining
    or refitting (as it's also called) every time it is used.
    """

    def __init__(self, bandwidth=None, radius_norm_step=100):
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
        self.radius_norm_step = radius_norm_step

    def fit(self, data):
        """Method to fit/train data and find clusters."""
        if self.bandwidth is None:
            all_data_centroid = np.average(data, axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.bandwidth = all_data_norm / self.radius_norm_step

        # find initial centroids which is each data point's location.
        centroids = {}
        for i in range(len(data)):
            centroids[i] = data[i]
        weights = [i for i in range(self.radius_norm_step)][::-1]

        # optimize centroids
        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                for featureset in data:
                    # check how far each datapoint is from current centroid.
                    distance = np.linalg.norm(featureset - centroid)
                    if distance == 0:
                        distance = 0.000000001
                    weight_index = int(distance / self.bandwidth)
                    if weight_index > self.radius_norm_step - 1:
                        weight_index = self.radius_norm_step - 1
                    to_add = (weights[weight_index]**2) * [featureset]
                    in_bandwidth += to_add
                new_centroid = np.average(in_bandwidth, axis=0)
                # make into tuple so I later can find unique arrays with set().
                new_centroids.append(tuple(new_centroid))
            uniques = sorted(set(new_centroids))
            # get rid of centroids that are close to eachother
            to_pop = []
            for i in uniques:
                for ii in uniques:
                    dist = np.linalg.norm(np.array(i) - np.array(ii))
                    if i == ii:
                        pass
                    elif dist <= self.bandwidth:
                        to_pop.append(ii)
                        break
            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass
            # further whittle down the centroids
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
        self.classifications = {}
        for i in range(len(self.centroids)):
            self.classifications[i] = []
        for featureset in data:
            distances = [np.linalg.norm(featureset - self.centroids[centroid])
                         for centroid in self.centroids]
            classification = distances.index(min(distances))
            self.classifications[classification].append(featureset)

        def predict(self, data):
            """Method to predict what cluster data belongs to."""
            distances = [np.linalg.norm(featureset - self.centroids[centroid])
                         for centroid in self.centroids]
            classification = distances.index(min(distances))
            return classification


# Create MeanShift instance and fit data
clf = MeanShift()
clf.fit(X)
centroids = clf.centroids
# Plot data
for classification in clf.classifications:
    color = colors[classification]
    for features in clf.classifications[classification]:
        plt.scatter(features[0], features[1], marker='o', color=color, s=100)
for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='x', s=150)
plt.show()
