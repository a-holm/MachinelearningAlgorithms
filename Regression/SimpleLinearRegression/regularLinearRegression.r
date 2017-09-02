# Simple linear regression for machine learning.
# 
# This file demonstrate knowledge of linear regression. By using
# conventional libraries.The idea of linear regression is to take continuous
# data and find the best fit of it to a line.
# 
# Simple linear regression just refers to the fact that the features only
# includes one column. So the label is composed by just one variable and one
# constant.

# Importing the data set
dataset = read.csv('Salary_Data.csv')

# Splitting the Dataset into a Training set and a Test set
# install.packages('caTools')
# library(caTools)
set.seed(123) # choose random number, only same number for debugging
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

# Fitting the Training set with Simple Linear regression model.
# Salary is propotional to YearsExperience and model built on training_set
regressor = lm(formula = Salary ~ YearsExperience, data = training_set)

# Predicting the Test set results
label_pred = predict(regressor, newdata = test_set)

# Visualising the regression line and training set
# install.packages("ggplot2")
# library(ggplot2)
ggplot() +
  geom_point(aes(x=training_set$YearsExperience, y=training_set$Salary),
             colour = 'red') +
  geom_line(aes(x=training_set$YearsExperience, y=predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Experience vs Salary (Training set)') +
  xlab('Years of experience') +
  ylab('Salary')


# Visualising the regression line and test set
ggplot() +
  geom_point(aes(x=test_set$YearsExperience, y=test_set$Salary),
             colour = 'red') +
  geom_line(aes(x=training_set$YearsExperience, y=predict(regressor, newdata = test_set)),
            colour = 'blue') +
  ggtitle('Experience vs Salary (test set)') +
  xlab('Years of experience') +
  ylab('Salary')
