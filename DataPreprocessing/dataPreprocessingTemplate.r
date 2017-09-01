# Data Preprocessing Template.
# 
# This is a file that I use as a template for data pre-processing in Machine
# learning projects. After a while I figured out it might be easier to have it as
# a reference.

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
