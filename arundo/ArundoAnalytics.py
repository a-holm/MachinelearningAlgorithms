# -*- coding: utf-8 -*-
"""Analysis file for the competency challenge given by Arundo Analytics.

This is a file that will do data preprocessing, test different relevant models,
compare the models and provide a basis for the analytics.

Usage:

        $ python ArundoAnalytics.py

Todo:
    *
"""
import inspect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (Imputer,
                                   LabelEncoder,
                                   OneHotEncoder,
                                   StandardScaler)
from sklearn.metrics import confusion_matrix
import statsmodels.formula.api as sm
import statsmodels.stats.api as sms
from sklearn.metrics import average_precision_score
from sklearn.metrics import r2_score
# regressors
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn import svm
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
# classifiers
"""from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier"""


def test_c_model(classifier, features, labels):
    """Function to test classifier model with random crossvalidation.

    Returns:
        (float) accuracy percentage
    """
    # Initial splitting of the Dataset into a Training set and a Test set
    feature_train, feature_test, label_train, label_test = train_test_split(
        features, labels, test_size=0.2)
    # fit the classifier.
    classifier.fit(feature_train, label_train)
    # Feature scaling, normalize scale is important. Especially on algorithms
    # involving euclidian distance.
    sc_feature = StandardScaler()
    feature_train = sc_feature.fit_transform(feature_train)
    feature_test = sc_feature.transform(feature_test)
    # Predicting the results of the Test set
    y_pred = classifier.predict(feature_test)
    # Creating the Confusion Matrix
    cm = confusion_matrix(label_test, y_pred)
    return float((cm[0][0] + cm[1][1] - cm[0][1] - cm[1][0]) / sum(sum(cm)))


def test_r_model(regressor, features, labels):
    """Function to test regressor model with random crossvalidation.

    Returns:
        (float) accuracy percentage
    """
    feature_train, feature_test, label_train, label_test = train_test_split(
        features, labels, test_size=0.2)
    regressor.fit(feature_train, label_train)
    # Feature scaling, normalize scale is important. Especially on algorithms
    # involving euclidian distance.
    sc_feature = StandardScaler()
    feature_train = sc_feature.fit_transform(feature_train)
    feature_test = sc_feature.transform(feature_test)
    # Predict the test set
    label_pred = regressor.predict(feature_test)
    # features = np.append(
    #     arr=np.ones(
    #         (np.shape(features)[0], 1)).astype(int), values=features, axis=1)
    # columnlist = list(range(features.shape[1]))  # liste med num rader
    # significant = 0.05
    # pre_r_value = 0
    # while True:
    #     features_opt = features[:, columnlist]
    #     regressor_OLS = sm.OLS(endog=labels, exog=features_opt).fit()
    #     pvalues = regressor_OLS.pvalues
    #     r_value = regressor_OLS.rsquared_adj
    #     if (np.max(pvalues) > significant):
    #         if (pre_r_value < r_value):
    #             break
    #         i = int(np.where(pvalues == np.max(pvalues))[0])
    #         columnlist.pop(i)
    #         pre_r_value = r_value
    #     else:
    #         break
    # features = features_opt
    # print(np.shape(features))
    # print(np.shape(labels))
    # plt.scatter(features, labels, color='r')
    # plt.plot(features, regressor.predict(features), color='b')
    # plt.title('Truth or Bluff (Regression Model)')
    # plt.xlabel('Position level')
    # plt.ylabel('Salary')
    # plt.show()


    # x_grid = np.arange(min(features), max(features), 0.1)
    # x_grid = x_grid.reshape((len(x_grid), 1))
    # plt.scatter(features, labels, color='r')
    # plt.plot(x_grid, regressor.predict(x_grid), color='b')
    # plt.title('Truth or Bluff (Decision Tree Regression)')
    # plt.xlabel('Position level')
    # plt.ylabel('Salary')
    # plt.show()
    return r2_score(label_test, label_pred)


def retrieve_name(var):
    """Retrieve name from variable."""
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, val in callers_local_vars if val is var]


if __name__ == '__main__':
    # importing the dataset
    dataset = pd.read_csv('Arundo_take_home_challenge_training_set.csv')
    # print(dataset.head())
    features = dataset.iloc[:, [1, 3, 4, 5, 6, 7]].values
    labels = dataset.iloc[:, 2].values

    # Encode categorical data and make them into numbers
    labelEncoder_x = LabelEncoder()
    features[:, 5] = labelEncoder_x.fit_transform(features[:, 5])
    # make dummy columns to avoid attributing order
    onehotencoder = OneHotEncoder(categorical_features=[5])
    features = onehotencoder.fit_transform(features).toarray()

    # Avoiding the Dummy variable trap
    features = features[:, 1:]

    # initialize regressor
    decisionTree = DecisionTreeRegressor()
    multipleRegression = LinearRegression()
    # poly_reg = PolynomialFeatures(degree=5)
    ran_forest_reg = RandomForestRegressor(n_estimators=310)
    svm_reg = SVR(kernel='rbf')
    regressorList = [decisionTree,
                     multipleRegression,
                     # poly_reg,
                     ran_forest_reg,
                     svm_reg]
    for regressor in regressorList:
        accuracyDistribution = np.array(
            [test_r_model(regressor, features, labels) for _ in range(100)])
        conf_interval = sms.DescrStatsW(accuracyDistribution).tconfint_mean()
        regressorName = retrieve_name(regressor)[0]
        aMean = np.mean(accuracyDistribution)
        sd = np.std(accuracyDistribution)
        print("{} has mean {} and 95% confidence interval".format(
            regressorName, aMean,) +
            " between [{}, {}] with ".format(
            conf_interval[0], conf_interval[1]) +
            "standard deviation: {}".format(sd))

    # initialize classifiers
"""    logReg = LogisticRegression()
    kNeighbors = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    svm_clasif = SVC(kernel='linear')
    kernel_svm = SVC(kernel='rbf')
    naive_bayes = GaussianNB()
    decisionTree = DecisionTreeClassifier(criterion='entropy')
    raForest = RandomForestClassifier(n_estimators=10, criterion='entropy')
    classifierList = [logReg,
                      kNeighbors,
                      svm_clasif,
                      kernel_svm,
                      naive_bayes,
                      decisionTree,
                      raForest]
    for classifier in classifierList:
        accuracyDistribution = np.array(
            [test_C_model(classifier, features, labels) for _ in range(100)])
        conf_interval = sms.DescrStatsW(accuracyDistribution).tconfint_mean()
        classifierName = retrieve_name(classifier)[0]
        aMean = np.mean(accuracyDistribution)
        sd = np.std(accuracyDistribution)
        print("{classifierName} has mean {aMean} and 95% confidence interval" +
            " between [{conf_interval[0]}, {conf_interval[1]}] with " +
            "standard deviation: {sd}")"""

dataset = pd.read_csv('Arundo_take_home_challenge_training_set.csv')
# print(dataset.head())
features = dataset.iloc[:, [1, 3, 4, 5, 6, 7]].values
labels = dataset.iloc[:, 2].values

# Encode categorical data and make them into numbers
labelEncoder_x = LabelEncoder()
features[:, 5] = labelEncoder_x.fit_transform(features[:, 5])
# make dummy columns to avoid attributing order
onehotencoder = OneHotEncoder(categorical_features=[5])
features = onehotencoder.fit_transform(features).toarray()

# Avoiding the Dummy variable trap
features = features[:, 1:]

# initialize regressor
decisionTree = DecisionTreeRegressor()
multipleRegression = LinearRegression()
poly_reg = PolynomialFeatures(degree=5)
ran_forest_reg = RandomForestRegressor(n_estimators=310)
svm_reg = SVR(kernel='rbf')
clf = SGDRegressor()
lasso = linear_model.Lasso(alpha=0.6)
elastic = ElasticNet()
ridge = linear_model.Ridge (alpha=.9)
svr = svm.SVR()
ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6),
                          n_estimators=300)
gradboost = GradientBoostingRegressor(loss='quantile',
                                n_estimators=250, max_depth=3,
                                learning_rate=.1, min_samples_leaf=9,
                                min_samples_split=9)
gradboostzero = GradientBoostingRegressor()
bagging = BaggingRegressor()
regressorList = [decisionTree,
                 multipleRegression,
                 # poly_reg,
                 ran_forest_reg,
                 svm_reg,
                 clf,
                 lasso,
                 elastic,
                 svr,
                 gradboostzero,
                 ada,
                 gradboost,
                 bagging
                 ]
for regressor in regressorList:
    accuracyDistribution = np.array(
        [float(test_r_model(regressor, features, labels)) for _ in range(100)])
    print(accuracyDistribution[0])
    conf_interval = sms.DescrStatsW(accuracyDistribution).tconfint_mean()
    regressorName = retrieve_name(regressor)[0]
    aMean = np.mean(accuracyDistribution)
    sd = np.std(accuracyDistribution)
    print("{} has mean {} and 95% confidence interval".format(
        regressorName, aMean,) +
        " between [{}, {}] with ".format(
        conf_interval[0], conf_interval[1]) +
        "standard deviation: {}".format(sd))


logReg = LogisticRegression()
kNeighbors = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
svm_clasif = SVC(kernel='linear')
kernel_svm = SVC(kernel='rbf')
naive_bayes = GaussianNB()
decisionTree = DecisionTreeClassifier(criterion='entropy')
raForest = RandomForestClassifier(n_estimators=10, criterion='entropy')
classifierList = [logReg,
                  kNeighbors,
                  svm_clasif,
                  kernel_svm,
                  naive_bayes,
                  decisionTree,
                  raForest]
for classifier in classifierList:
    accuracyDistribution = np.array(
        [test_c_model(classifier, features, labels) for _ in range(100)])
    conf_interval = sms.DescrStatsW(accuracyDistribution).tconfint_mean()
    classifierName = retrieve_name(classifier)[0]
    aMean = np.mean(accuracyDistribution)
    sd = np.std(accuracyDistribution)
    print("{} has mean {} and 95% confidence interval".format(
    regressorName, aMean,) +
    " between [{}, {}] with ".format(
    conf_interval[0], conf_interval[1]) +
    "standard deviation: {}".format(sd))