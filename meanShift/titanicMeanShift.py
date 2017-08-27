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
