# -*- coding: utf-8 -*-
"""Support Vector Machine (SVM) classification for machine learning.

SVM is a binary classifier. The objective of the SVM is to find the best
separating hyperplane in vector space which is also referred to as the
decision boundary. And it decides what separating hyperplane is the 'best'
because the distance from it and the associating data it is separating is the
greatest at the plane in question.

This is the file where I create the algorithm from scratch.

dataset is breast cancer data from: http://archive.ics.uci.edu/ml/datasets.html

Example:

        $ python howItWorksSupportVectorMachine.py

Todo:
    * Sketch out the framework
"""

# minimize magnitude(w) and maximize b
# with constraint y_i*(x_i*w+b)>=1
# or Class*(KnownFeatures*w+b)>=1
