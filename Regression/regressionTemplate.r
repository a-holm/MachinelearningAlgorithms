# Regression template for machine learning.

# Importing the data set
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the Dataset into a Training set and a Test set
# install.packages('caTools')
# library(caTools)
set.seed(123) # choose random number, only same number for debugging
split = sample.split(dataset$Salary, SplitRatio = 0.8)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

# Feature Scaling
training_set = scale(training_set)
test_set = scale(test_set)

# Fit regression to the dataset
# Create regressor

# Predict a new result with regression model
y_pred = predict(regressor, newdata = data.frame(Level = 6.5))

# Visualize graphic result of the regression model
# install.packages('ggplot2')
# library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour='red') +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
            colour='blue') +
  ggtitle('Truth or Bluff (Regression Model)') +
  xlab('Level') + ylab('Salary')

# HIGH RESOLUTION: Visualize graphic result of the regression model
# install.packages('ggplot2')
# library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour='red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level=x_grid))),
            colour='blue') +
  ggtitle('Truth or Bluff (Regression Model)') +
  xlab('Level') + ylab('Salary')
