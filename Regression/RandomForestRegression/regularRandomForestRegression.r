# Random Forest Regression for machine learning.
# 
# Random forest algorithm is a supervised classification algorithm. As the name
# suggest, this algorithm creates the forest with a number of decision trees.
# 
# In general, the more trees in the forest the more robust the forest looks like.
# In the same way in the random forest classifier, the higher the number of trees
# in the forest gives the high accuracy results.
# Importing the data set

dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Fit Random Forest Regression to the dataset
# install.packages("randomForest")
# library(randomForest)
set.seed(1234) # For debuging
regressor = randomForest(x = dataset[1], y = dataset$Salary,
                         ntree = 100)

# Predict a new result with Random Forest Regression model
y_pred = predict(regressor, newdata = data.frame(Level = 6.5))

# HIGH RESOLUTION: Visualize graphic result of the Random Forest Regression model
# install.packages('ggplot2')
# library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour='red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level=x_grid))),
            colour='blue') +
  ggtitle('Truth or Bluff (Random Forest Regression)') +
  xlab('Level') + ylab('Salary')
