# Data Preprocessing Template.
# 
# This is a file that I use as a template for data pre-processing in Machine
# learning projects. After a while I figured out it might be easier to have it as
# a reference. And to save time.

# Importing the data set
dataset = read.csv('data/Data.csv')

# Dealing with missing data
dataset$Age = ifelse(is.na(dataset$Age), ave(dataset$Age,
                     FUN = function(x) mean(x, na.rm = TRUE)), dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary), ave(dataset$Salary,
                        FUN = function(x) mean(x, na.rm = TRUE)), dataset$Salary)

# Encode categorical data and make them into numeric categories.
dataset$Country = factor(dataset$Country, levels = c('France', 'Spain', 'Germany'),
                         labels = c(1,2,3))
dataset$Purchased = factor(dataset$Purchased,
                           levels = c('No', 'Yes'),
                           labels = c(0,1))

# Splitting the Dataset into a Training set and a Test set
# install.packages('caTools')
# library(caTools)
set.seed(123) # choose random number, only same number for debugging
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)


# Feature scaling, normalize scale is important. Especially on algorithms 
# involving euclidian distance. Two main feature scaling formulas are:
# Standardisation: x_stand = (x-mean(x))/(standard_deviation(x))
# Normalisation: x_norm = (x-min(x))/(max(x)-min(x))
training_set[, 2:3] = scale(training_set[, 2:3])
test_set[, 2:3] = scale(test_set[, 2:3])
