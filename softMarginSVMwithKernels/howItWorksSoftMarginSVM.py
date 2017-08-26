# -*- coding: utf-8 -*-
"""Soft Margin SVM classification with kernels for machine learning.

Soft margin SVM is basically an SVM (see folder **supportVectorMachine**) which
has some 'slack' and allows features to be 'wrongly' classified to avoid
overfitting the classifier. This also includes kernels. Kernels use the inner
product to help us transform the feature space to make it possible for Support
Vector Machines to create a good hyperplane with non-linear feature sets.

I basically just do the 'from scratch' in this part because all this can easily
be done by just adding some parameters to sklearn's svm.SVC().

Example:

        $ python howItWorksSoftMarginSVM.py.py

Todo:
    *
"""
import numpy as np
from numpy import linalg
# Because I made a convex solver in 'howItWorksSupportVectorMachine.py' I will
# just use a library for it now because it's simpler.
import cvxopt
import cvxopt.solvers


def linear_kernel(x1, x2):
    """Linear kernel function.

    if this kernel is used then the decision boundary hyperplane will have a
    linear form.
    """
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=3):
    """Polynomial kernel function.

    if this kernel is used then the decision boundary hyperplane will have a
    Polynomial form.
    """
    return (1 + np.dot(x, y))**p


def gaussian_kernel(x, y, sigma=5.0):
    """Gaussian kernel function.

    if this kernel is used then the decision boundary hyperplane will have a
    Gaussian form.
    """
    return np.exp(-linalg.norm(x - y)**2 / (2 * (sigma**2)))


class SVM(object):
    """Support Vector Machine (SVM) class.

    This class is for creating an instance of a SVM. To avoid retraining or
    refitting (as it's also called) every time it is used.
    """

    def __init__(self, kernel=linear_kernel, C=None):
        """The __init__ method of the SVM class.

        Args:
            kernel (function name): The kernel that will be used.
        Default linear kernel.
            C: the max sum of all the distances of the features that are
        wrongly classified during fitting/training. Default is 'None', if C is
        None then it's a hard margin SVM with no slack.
        """
        self.kernel = kernel
        self.C = C
        if self.C is not None:
            self.C = float(self.C)
