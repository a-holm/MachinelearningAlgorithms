# -*- coding: utf-8 -*-
"""Linear regression for machine learning

This file demonstrate knowledge of linear regression. By using
building the algorithm from scratch.The idea of linear regression is to take
continuous data and find the best fit of it to a line.
"y=mx+b" is the equation of a line, and we need to figure out the
best slope (m) and y-intercept (b) to fit the line best.

This is for simple 2d regression.

Example:

        $ python howItWorksLinearRegression.py

Todo:
    *
"""
from statistics import mean

from matplotlib import style
import matplotlib.pyplot as plt

import numpy as np

style.use('fivethirtyeight')

simple_x = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
simple_y = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


def best_fit_slope_and_intercept(xs, ys):
    # Best fit m(slope)
    m = (mean(xs) * mean(ys) - mean(xs * ys)) / (mean(xs)**2 - mean(xs**2))
    # Best fit y(intercept)
    b = mean(ys) - m * mean(xs)
    return m, b

m, b = best_fit_slope_and_intercept(simple_x, simple_y)
regression_line = [(m * x) + b for x in simple_x]  # "y=mx+b"

predict_x = 8
predict_y = (m * predict_x) + b

plt.scatter(simple_x, simple_y)
plt.scatter(predict_x, predict_y, color='r')
plt.plot(simple_x, regression_line, color='g')
plt.show()
