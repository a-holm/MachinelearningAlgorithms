# Regression algorithms and projects
Python examples of Regression. Each algorithm has it's own folder.

By making the algorithms from scratch I hope to show that I do not only know how to use the algorithms, but I also understand how it functions. By using conventional libraries for machine learning I hope to show that I can be effective and take advantage of the high-performance backend of libraries like for example, Scikit-learn and Tensorflow.

Folder names are **bolded** while filenames are `highlighted` for readability.

## FOLDER CONTENTS (in alphabetical order):

### **_DecisionTreeRegression_**
Currently empty, check back soon for updates.

### **_MultipleLinearRegression_**
A linear regression model that contains more than one predictor variable is called a multiple linear regression model. It is basically the same as Simple Linear regression, but with more predictor variables (features). The idea is that linearly related predictor variables can approximate the labels with a 'best fitted' hyperplane or surface. The model assumes:
  1. Linearity
  2. Homoscedasticity
  3. Multivariate normality
  4. Independence of errors
  5. Lack of multicollinearity

  * `50_Startups.csv` - Data used in `regularMultipleRegression.py` and `regularMultipleRegression.r`.
  * `regularMultipleRegression.py` - Multiple Linear Regression in Python.
  * `regularMultipleRegression.r` - Multiple Linear Regression in R.

### **_PolynomialRegression_**
polynomial regression is a form of regression analysis in which the relationship between the independent variable x and the dependent variable y is modelled as an nth degree polynomial in x. Polynomial regression fits a nonlinear relationship between the value of x and the corresponding conditional mean of y, denoted E(y |x)Although polynomial regression fits a nonlinear model to the data, as a statistical estimation problem it is linear, in the sense that the regression function E(y | x) is linear in the unknown parameters that are estimated from the data. For this reason, polynomial regression is considered to be a special case of multiple linear regression.

  * `Position_Salaries.csv` - Data used in `regularPolynomialRegression.py` and `regularPolynomialRegression.r`.
  * `regularPolynomialRegression.py` - Polynomial Regression in Python.
  * `regularPolynomialRegression.r` - Polynomial Regression in R.

### **_RandomForestRegression_**
Currently empty, check back soon for updates.

### **_SimpleLinearRegression_**
The linear regression is a way to model linear data and thereby be able to predict values (or 'labels' as they are called in machine learning) based on the predictor variables (features). The idea is that linear data can be approximated well with a 'best fitted' line. The model assumes:
  1. Linearity
  2. Homoscedasticity
  3. Multivariate normality
  4. Independence of errors
  5. Lack of multicollinearity

  * `howItWorksLinearRegression.py` - The algorithm coded *from scratch*.
  * `linearregression.pickle` - The trained data saved as a pickle file to save computing time. (see `regularLinearRegression.py`)
  * `regularLinearRegression.py` - The algorithm coded with Scikit-learn (python library for machine learning).
  * `regularLinearRegression.r` - Simple linear regression in R.
  * `regularLinearRegression2.py` - Different variation of it using my new data preprocessing template.
  * `Salary_Data.csv` - Data used in `regularLinearRegression2.py` and `regularLinearRegression.r`.


### **_SupportVectorRegression_**
Support Vector Machine can also be used as a regression method, maintaining all
the main features that characterize the algorithm (maximal margin). The Support
Vector Regression (SVR) uses the same principles as the SVM for classification,
with only a few minor differences. First of all, because output is a real
number it becomes very difficult to predict the information at hand, which has
infinite possibilities. In the case of regression, a margin of tolerance is set
in approximation to the SVM which would have already requested from the
problem. But besides this fact, there is also a more complicated reason, the
algorithm is more complicated therefore to be taken in consideration. However,
the main idea is always the same: to minimize error, individualizing the
hyperplane which maximizes the margin, keeping in mind that part of the error
is tolerated.

  * `Position_Salaries.csv` - Data used in `regularSVMRegression.py` and `regularSVMRegression.r`.
  * `regularSVMRegression.py` - SVM Regression in Python.
  * `regularSVMRegression.r` - SVM Regression in R.
