# Polynomial regression for machine learning.
# 
# polynomial regression is a form of regression analysis in which the
# relationship between the independent variable x and the dependent variable y is
# modelled as an nth degree polynomial in x. Polynomial regression fits a
# nonlinear relationship between the value of x and the corresponding conditional
# mean of y, denoted E(y |x)Although polynomial regression fits a nonlinear model
# to the data, as a statistical estimation problem it is linear, in the sense
# that the regression function E(y | x) is linear in the unknown parameters that
# are estimated from the data. For this reason, polynomial regression is
# considered to be a special case of multiple linear regression.


# Importing the data set
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# no need to split
# # Splitting the Dataset into a Training set and a Test set
# # install.packages('caTools')
# # library(caTools)
# set.seed(123) # choose random number, only same number for debugging
# split = sample.split(dataset$Salary, SplitRatio = 0.8)
# training_set = subset(dataset, split==TRUE)
# test_set = subset(dataset, split==FALSE)

# Fit Polynomial regression to the dataset
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
dataset$Level5 = dataset$Level^5
poly_reg = lm(formula = Salary ~ ., data = dataset)
summary(poly_reg)

# Visualize graphic result of Polynomial regression
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour='red') +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
            colour='blue') +
  xlab('Level') + ylab('Salary')

# Predict a new result with Polynomial regression
y_pred = predict(poly_reg, newdata = data.frame(Level = 6.5,
                                                Level2 = 6.5^2,
                                                Level3 = 6.5^3,
                                                Level4 = 6.5^4,
                                                Level5 = 6.5^5))
