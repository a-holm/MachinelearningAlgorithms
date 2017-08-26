# -*- coding: utf-8 -*-
"""Support Vector Machine (SVM) classification for machine learning.

SVM is a binary classifier. The objective of the SVM is to find the best
separating hyperplane in vector space which is also referred to as the
decision boundary. And it decides what separating hyperplane is the 'best'
because the distance from it and the associating data it is separating is the
greatest at the plane in question. The SVM classifies as either on the positive
or negative side of the hyperplane.

This is the file where I create the algorithm from scratch. This is algorithm
for 2D data.

Example:

        $ python howItWorksSupportVectorMachine.py

Todo:
    *
"""
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')


class SupportVectorMachine:
    """Support Vector Machine (SVM) class.

    This class is for creating an instance of a SVM. To avoid retraining or
    refitting (as it's also called) it all the time.
    """

    def __init__(self, visualization=True):
        """The __init__ method of the SupportVectorMachine class.

        Args:
            visualization (bool): Default set to True due to debugging
        """
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'c'}
        if visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    def fit(self, data):
        """Method to train the SVM object as a convex optimization problem.

        Uses a simple method to solve convex optimization problems. There are
        more sophisticated methods, but for the sake of simplicity (because I
        want to do it from scratch) I use a simpler method.

        Return:
            (int)

        Args:
            data (:obj:`list` of :obj:`int`:): Data to be fitted/trained.
        """
        self.data = data

        opt_dict = {}  # will be populated like { ||w||: [w,b]}

        # Everytime we get a value we will transform them by these multipliers
        # to see what works on each step we 'step down' the convex area.
        transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]

        # all_data is used to find max and min values.
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None  # to avoid holding it in memory

        # starting values for training algorithm
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001]
        # b is expensive and does not need to be as precise.
        b_range_multiple = 5
        # we do not need to take as small steps with b as we do with w.
        # we could do it, but it would make it too complex for just a portfolio
        b_multiple = 5
        # to cut a few memory corners all w = [latest_optimum, latest_optimum]
        optimum_multiplier = 10
        latest_optimum = self.max_feature_value * optimum_multiplier

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False  # We can use this because convex problem.
            while not optimized:  # following code can probably be threaded
                for b in np.arange(
                    -1 * (self.max_feature_value * b_range_multiple),
                    self.max_feature_value * b_range_multiple,
                        step * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True
                        """
                        Here is the point that could probably be speeded up
                        this is the weakest part of the algorithm. I have
                        more algoritms to show of so I am not going to use
                        time to make this more effective. There are libraries
                        that are more effective anyway.
                        """
                        # TODO: Possibly add a break later
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                if w[0] < 0:
                    optimized = True
                    print('Optimized a step')
                else:
                    w = w - step
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2

    def predict(self, features):
        """Method to predict features based on the SVM object.

        Return:
            (int) an element-wise indication of the classification of the
        features.

        Args:
            features (:obj:`list` of :obj:`int`:): Features to be predicted.
        """
        # just see if (x.w+b) is negative or positive
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], marker="*",
                            s=80, c=self.colors[classification])
        return classification

    def visualize(self):
        """Method for visualization and plotting."""
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i])
          for x in data_dict[i]] for i in data_dict]

        def hyperplane(x, w, b, v):  # hyperplane v = x.w+b
            """Method to return values of hyperplanes for visualization.

            Args:
                x, w, b (int): values to figure out the hyperplane = x.w+b
                v (int): value of the hyperplane, either the positive support
                        vector (1), the negative support vector (-1) or the
                        decision boundary (0).
            """
            return (-w[0] * x - b + v) / w[1]
        datarng = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyp_x_min = datarng[0]
        hyp_x_max = datarng[1]

        # plot positive support vector hyperplane. (w.x+b) = 1
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        # plot negative support vector hyperplane. (w.x+b) = -1
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # plot decision boundary hyperplane. (w.x+b) = 0
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()

# get training data
negative_array = np.array([[1, 7], [2, 8], [3, 8]])
positive_array = np.array([[5, 1], [6, -1], [7, 3]])
data_dict = {-1: negative_array, 1: positive_array}

svm = SupportVectorMachine()
svm.fit(data=data_dict)

# prediction data and prediction
pred_test = [[0, 10], [1, 3], [3, 4], [3, 5], [5, 5], [5, 6], [6, -5], [5, 8]]
for p in pred_test:
    svm.predict(p)

svm.visualize()
