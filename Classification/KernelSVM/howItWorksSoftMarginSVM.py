# -*- coding: utf-8 -*-
"""Soft Margin SVM classification with kernels for machine learning.

Soft margin SVM is basically an SVM (see folder **supportVectorMachine**) which
has some 'slack' and allows features to be 'wrongly' classified to avoid
overfitting the classifier. This also includes kernels. Kernels use the inner
product to help us transform the feature space to make it possible for Support
Vector Machines to create a good hyperplane with non-linear feature sets.

This file can basically do the same as the "from scratch" algorithm in folder
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

if __name__ == '__main__':

    def gen_lin_seperable_data():
        """Function to generate linearly seperable 2d training data."""
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def gen_non_lin_seperable_data():
        """Function to generate non-linear seperable training data."""
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0,0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def gen_lin_seperable_overlap_data():
        """Function to generate linearly seperable overlapping data.

        Function to generate linearly seperable, but overlapping 2d training
        data. This is training data that needs a soft margin
        """
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def split_train(X1, y1, X2, y2):
        """Function to split the training data."""
        X1_train = X1[:90]
        y1_train = y1[:90]
        X2_train = X2[:90]
        y2_train = y2[:90]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train

    def split_test(X1, y1, X2, y2):
        """Function to split the testing data."""
        X1_test = X1[90:]
        y1_test = y1[90:]
        X2_test = X2[90:]
        y2_test = y2[90:]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test

    def test_linear():
        """Function to test linear kernel."""
        X1, y1, X2, y2 = gen_lin_seperable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM()  # because it's convention to use clf
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))

    def test_non_linear():
        """Function to test polynomial kernel."""
        X1, y1, X2, y2 = gen_non_lin_seperable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM(polynomial_kernel)  # because it's convention to use clf
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))

    def test_soft():
        """Function to test linear kernel with soft margin."""
        X1, y1, X2, y2 = gen_lin_seperable_overlap_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM(C=1000.1)  # because it's convention to use clf
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))

    # test_linear()
    test_non_linear()
    # test_soft()
