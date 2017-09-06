# Support Vector Machine (SVM) classification for machine learning.
# 
# SVM is a binary classifier. The objective of the SVM is to find the best
# separating hyperplane in vector space which is also referred to as the
# decision boundary. And it decides what separating hyperplane is the 'best'
# because the distance from it and the associating data it is separating is the
# greatest at the plane in question.

# Importing the data set
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[, 3:5]

# Splitting the Dataset into a Training set and a Test set
# install.packages('caTools')
# library(caTools)
set.seed(123) # choose random number, only same number for debugging
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)


# Feature scaling, normalize scale is important. Especially on algorithms 
# involving euclidian distance. Two main feature scaling formulas are:
# Standardisation: x_stand = (x-mean(x))/(standard_deviation(x))
# Normalisation: x_norm = (x-min(x))/(max(x)-min(x))
training_set[, 1:2] = scale(training_set[, 1:2])
test_set[, 1:2] = scale(test_set[, 1:2])


# Fitting the Classifier Model to the training_set
# install.packages('e1071')
# library(e1071)
classifier = svm(formula = Purchased ~ ., data = training_set,
                 type = 'C-classification', kernel = 'linear')
  
  
# Predict the test_set results by using the classifier
y_pred = predict(classifier, newdata = test_set[-3])


# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)


# Plot the training_set results
# install.packages('ElemStatLearn')
# library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set>0.5, 1, 0)
plot(set[, -3],
     main = 'SVM Classifier (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))


# Plot the test_set results
# install.packages('ElemStatLearn')
# library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set>0.5, 1, 0)
plot(set[, -3],
     main = 'SVM Classifier (testing set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

