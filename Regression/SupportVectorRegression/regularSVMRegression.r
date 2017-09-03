# Support Vector regression for machine learning.
# 
# Support Vector Machine can also be used as a regression method, maintaining all
# the main features that characterize the algorithm (maximal margin). The Support
# Vector Regression (SVR) uses the same principles as the SVM for classification,
# with only a few minor differences. First of all, because output is a real
# number it becomes very difficult to predict the information at hand, which has
# infinite possibilities. In the case of regression, a margin of tolerance is set
# in approximation to the SVM which would have already requested from the
# problem. But besides this fact, there is also a more complicated reason, the
# algorithm is more complicated therefore to be taken in consideration. However,
# the main idea is always the same: to minimize error, individualizing the
# hyperplane which maximizes the margin, keeping in mind that part of the error
# is tolerated.

# Importing the data set
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the Dataset into a Training set and a Test set
# install.packages('caTools')
# library(caTools)
# set.seed(123) # choose random number, only same number for debugging
# split = sample.split(dataset$Salary, SplitRatio = 0.8)
# training_set = subset(dataset, split==TRUE)
# test_set = subset(dataset, split==FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fit SVR to the dataset
# install.packages('e1071')
# library(e1071)
regressor = svm(formula = dataset$Salary ~ .,
                data = dataset,
                type = 'eps-regression')

# Predict a new result with SVR model
y_pred = predict(regressor, newdata = data.frame(Level = 6.5))

# Visualize graphic result of the SVR model
# install.packages('ggplot2')
# library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour='red') +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
            colour='blue') +
  ggtitle('Truth or Bluff (SVR)') +
  xlab('Level') + ylab('Salary')