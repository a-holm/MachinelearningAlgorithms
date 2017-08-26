# -*- coding: utf-8 -*-
"""Support Vector Machine (SVM) classification for machine learning.

SVM is a binary classifier. The objective of the SVM is to find the best
separating hyperplane in vector space which is also referred to as the
decision boundary. And it decides what separating hyperplane is the 'best'
because the distance from it and the associating data it is separating is the
greatest at the plane in question. The SVM classifies as either on the positive
or negative side of the hyperplane.

This is the file where I create the algorithm from scratch.

dataset is breast cancer data from: http://archive.ics.uci.edu/ml/datasets.html

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
        """Docstring on the __init__ method.

        Args:
            visualization (bool): Default set to True due to debugging
        """
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
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

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001]

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

        return classification


# get training data
negative_array = np.array([[1, 7], [2, 8], [3, 8]])
positive_array = np.array([[5, 1], [6, -1], [7, 3]])
data_dict = {-1: negative_array, 1: positive_array}
