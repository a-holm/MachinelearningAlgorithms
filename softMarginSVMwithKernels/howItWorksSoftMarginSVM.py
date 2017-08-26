# -*- coding: utf-8 -*-
"""Soft Margin SVM classification with kernels for machine learning.

Soft margin SVM is basically an SVM (see folder **supportVectorMachine**) which
has some 'slack' and allows features to be 'wrongly' classified to avoid
overfitting the classifier. This also includes kernels. Kernels use the inner
product to help us transform the feature space to make it possible for Support
Vector Machines to create a good hyperplane with non-linear feature sets.

I basically just do the 'from scratch' in this part because all this can easily
be done by just adding some parameters to sklearn's svm.SVC(). This file can
basically do the same as the "from scratch" algorithm in folder
"supportVectorMachine", but this is much more complex to account for margins
and more dimensions involved. This also involves more complex math, matrix
algebra and Lagrange multipliers.

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

    def fit(self, X, y):
        """Method to train the SVM object as a convex optimization problem.

        Return:
            (void)

        Args:
            X (np.array): the features
            y (np.array): the labels
        """
        n_samples, n_features = X.shape

        # Creating all the values for the quadratic Programming solver
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve Quadratic Programming problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero Lagrange multipliers
        sv = a > 1e-5  # due to floating point errors
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # find the Intercept/ bias b
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)

        # find the Weight vector w
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

        def project(self, X):
            """Method is useful for getting the prediction depending on kernel.

            Return:
                (int or np.array of ints) A number which indicate the
            classification of the features by being positive or negative.
            """
            if self.w is not None:
                return np.dot(X, self.w) + self.b
            else:
                y_predict = np.zeros(len(X))
                for i in range(len(X)):
                    s = 0
                    for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                        s += a * sv_y * self.kernel(X[i], sv)
                    y_predict[i] = s
                return y_predict + self.b

        def predict(self, X):
            """Method to predict features X."""
            return np.sign(self.project(X))

