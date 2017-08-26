# Machine learning and Artificial intelligence projects in Python
Python examples of machine learning algorithms. All examples are shown by using conventional libraries for machine learning and by also building the algorithms from scratch.
Each algorithm has it's own folder.

By making the algorithms from scratch I hope to show that I do not only know how to use the algorithms, but I also understand how it functions. By using conventional libraries for machine learning I hope to show that I can be effective and take advantage of the high-performance backend of libraries like for example, Scikit-learn and Tensorflow.

## CONTENTS (in alphabetical order):

### **_kMeans_** 
K-means clustering is a unsupervised method to cluser or group the data. K-means allows you to choose the number (k) of categories/groups and categorizes it automatically when it has come up with solid categories.

  * `howItWorksKMeans.py` - The algorithm coded from scratch.
  * `regularKMeans.py` - The algorithm coded with Scikit-learn (python library for machine learning).

### **_kNearestNeighbors_** 
The k-nearest neighbors algorithm is a method used for classification and regression. The idea of K Nearest Neighbors classification is to best divide and separate the data based on clustering the data and to classify based on the proximity to it's K closest neighbors and their classifications. Where 'k' is the number of neighbors that are involved in the classification.

  * `breast-cancer-wisconsin.data` - Data used in the example.
  * `breast-cancer-wisconsin.names` - Information about the data used in the example.
  * `howItWorksKNearestNeighbors.py` - The algorithm coded from scratch.
  * `regularNearestNeighbors.py` - The algorithm coded with Scikit-learn (python library for machine learning).

### **_linearRegression_** 
The linear regression is a way to model linear data and thereby be able to predict values (or 'labels' as they are called in machine learning) based on the features. The idea is that linear data can be approximated well with a 'best fitted' line.

  * `howItWorksLinearRegression.py` - The algorithm coded from scratch.
  * `linearregression.pickle` - The trained data saved as a pickle file to save computing time. (see *regularLinearRegression.py*)
  * `regularLinearRegression.py` - The algorithm coded with Scikit-learn (python library for machine learning).

### **_softMarginSVMwithKernels_** 
Soft margin SVM is basically an SVM (see folder **supportVectorMachine**) which has some 'slack' and allows features to be 'wrongly' classified to avoid overfitting the classifier. This also includes kernels. Kernels use the inner product to help us transform the feature space to make it possible for Support Vector Machines to create a good hyperplane with non-linear feature sets.

This can basically do the same as the "from scratch" algorithm in folder **supportVectorMachine**, but this is much more complex to account for margins and more dimensions involved.

I also added methods to check and predict non-linear data.

  * `breast-cancer-wisconsin.data` - Data used in the example.
  * `howItWorksSoftMarginSVM.py` - The algorithm coded from scratch.
  * `regularSoftMarginSVM.py` - The algorithm coded with Scikit-learn (python library for machine learning).


### **_supportVectorMachine_** 
Support vector machines (SVMs) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. The objective of the SVM is to find the best separating hyperplane in vector space which is also referred to as the decision boundary.

The algorithm coded from scratch is a bare-bones and simple implementation to mainly show the thinking clearly without adding a lot of other elements. This algorithm can only take linearly separable data that does not overlap. For a more advanced implementation with overlapping and non-linear data see the folder **softMarginSVMwithKernels**.

  * `breast-cancer-wisconsin.data` - Data used in the example.
  * `howItWorksSupportVectorMachine.py` - The algorithm coded from scratch.
  * `regularSupportVectorMachine.py` - The algorithm coded with Scikit-learn (python library for machine learning).

