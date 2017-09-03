# Decision Tree regression for machine learning.
# 
# Decision tree builds regression or classification models in the form of a tree
# structure. It brakes down a dataset into smaller and smaller subsets while at
# the same time an associated decision tree is incrementally developed. The final
# result is a tree with decision nodes and leaf nodes.

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

# Fit Decision Tree Regression to the dataset
# install.packages('rpart')
# library(rpart)
regressor = rpart(formula = dataset$Salary ~ .,
                  data = dataset,
                  control = rpart.control(minsplit = 1))

# Predict a new result with Decision Tree Regression model
y_pred = predict(regressor, newdata = data.frame(Level = 6.5))

# HIGH RESOLUTION: Visualize graphic result of the Decision Tree Regression model
# install.packages('ggplot2')
# library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour='red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level=x_grid))),
            colour='blue') +
  ggtitle('Truth or Bluff (Decision Tree Regression)') +
  xlab('Level') + ylab('Salary')
