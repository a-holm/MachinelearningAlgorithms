# Multiple linear regression for machine learning.
# 
# A linear regression model that contains more than one predictor variable is
# called a multiple linear regression model. It is basically the same as Simple
# Linear regression, but with more predictor variables (features). The idea is
# that linearly related predictor variables can approximate the labels with a
# 'best fitted' hyperplane or surface.

# Importing the data set
dataset = read.csv('50_Startups.csv')

# Encode categorical data and make them into numeric categories.
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1,2,3))

# Splitting the Dataset into a Training set and a Test set
# install.packages('caTools')
# library(caTools)
set.seed(123) # choose random number, only same number for debugging
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

# Don't need Feature Scaling

# Fit the training set with the mutliple linear regression model
regressor = lm(formula = Profit ~ ., data = training_set)
summary(regressor)
# And from it we can see
regressor = lm(formula = Profit ~ R.D.Spend, data = training_set)

# Predict Test set results
label_pred = predict(regressor, newdata = test_set)