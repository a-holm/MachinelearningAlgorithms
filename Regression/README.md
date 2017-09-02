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

  * `50_Startups.csv` - Multiple Linear Regression
  * `regularMultipleRegression.py` - Multiple Linear Regression in Python.
  * `regularMultipleRegression.r` - Multiple Linear Regression in R.

### **_PolynomialRegression_**
Currently empty, check back soon for updates.

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
Currently empty, check back soon for updates.