# Machine learning Algorithms in Python
Python examples of machine learning algorithms. All examples are shown by using conventional libraries and by also building the algorithms from scratch.
Each algorithm has it's own folder.

## CONTENTS:

### **_kNearestNeighbors_** 
The k-nearest neighbors algorithm is a method used for classification and regression. The idea of K Nearest Neighbors classification is to best divide and separate the data based on clustering the data and to classify based on the proximity to it's K closest neighbors and their classifications. Where 'k' is the number of neighbors that are involved in the classification.

  * `breast-cancer-wisconsin.data` - Data used in the example.
  * `breast-cancer-wisconsin.names` - Information about the data used in the example.
  * `howItWorksKNearestNeighbors.py` - The algorithm coded from scratch.
  * `regularNearestNeighbors.py` - The algorithm coded with python libraries.

### **_linearRegression_** 
The linear regression is a way to model linear data and thereby be able to predict values (or 'labels' as they are called in machine learning) based on the features. The idea is that linear data can be approximated well with a 'best fitted' line.

  * `howItWorksLinearRegression.py` - The algorithm coded from scratch.
  * `linearregression.pickle` - The trained data saved as a pickle file to save computing time. (see *regularLinearRegression.py*)
  * `regularLinearRegression.py` - The algorithm coded with python libraries.
