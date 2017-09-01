# -*- coding: utf-8 -*-
"""Linear regression for machine learning

This file demonstrate knowledge of linear regression. By building the algorithm
from scratch.The idea of linear regression is to take continuous data and find
the best fit of it to a line.
"y=mx+b" is the equation of a line, and we need to figure out the
best slope (m) and y-intercept (b) to fit the line best.

This is for simple 2d regression.

Example:

        $ python howItWorksLinearRegression.py

Todo:
    *
"""
import random

from statistics import mean

from matplotlib import style
import matplotlib.pyplot as plt

import numpy as np

style.use('fivethirtyeight')

# simple_x = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
# simple_y = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


def create_dataset(howmany, variance, step=2, correlation=False):
    # for test-data
    val = 1
    ys = []
    for i in range(howmany):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

simple_x, simple_y = create_dataset(40, 10, 2, correlation='pos')


def best_fit_slope_and_intercept(xs, ys):
    # Best fit m(slope)
    m = (mean(xs) * mean(ys) - mean(xs * ys)) / (mean(xs)**2 - mean(xs**2))
    # Best fit y(intercept)
    b = mean(ys) - m * mean(xs)
    return m, b


def squared_error(ys_orig, ys_line):
    # Figuring out the squared_error that we need to calculate R^2
    return sum((ys_line - ys_orig)**2)


def coefficient_of_determination(ys_orig, ys_line):
    # Figuring out R^2 = 1 - (squared_error(y_hat) / squared_error(mean(y)))
    y_mean_line = [mean(ys_orig) for y in ys_orig]  # line of mean(y)s
    squared_error_regressionline = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regressionline / squared_error_y_mean)

m, b = best_fit_slope_and_intercept(simple_x, simple_y)
regression_line = [(m * x) + b for x in simple_x]  # "y=mx+b"

predict_x = 8
predict_y = (m * predict_x) + b
r_squared = coefficient_of_determination(simple_y, regression_line)

print(r_squared)

plt.scatter(simple_x, simple_y)
plt.plot(simple_x, regression_line, color='g')
plt.scatter(predict_x, predict_y, color='r')
plt.show()
