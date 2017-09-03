# Classification algorithms and projects
Python examples of Classification. Each algorithm has it's own folder.

By making the algorithms from scratch I hope to show that I do not only know how to use the algorithms, but I also understand how it functions. By using conventional libraries for machine learning I hope to show that I can be effective and take advantage of the high-performance backend of libraries like for example, Scikit-learn and Tensorflow.

Folder names are **bolded** while filenames are `highlighted` for readability.

## FOLDER CONTENTS (in alphabetical order):


  - `classificationTemplate.py` This is a template file for classifications in Python.
  - `classificationTemplate.r` This is a template file for classifications in R.
  - `requirements.txt` Python requirements to run all files in the classification folder.


### **_DecisionTreeClassification_**
Currently empty, check back soon for updates.

### **_KernelSVM_** 
Kernel SVM is basically an SVM (see folder **SupportVectorMachine**) which has some 'slack' and allows features to be 'wrongly' classified to avoid overfitting the classifier. This also includes kernels. Kernels use the inner product to help us transform the feature space to make it possible for Support Vector Machines to create a good hyperplane with non-linear feature sets.

I also added methods to check and predict non-linear data.

  * `breast-cancer-wisconsin.data` - Data used in some of the examples.
  * `howItWorksSoftMarginSVM.py` - The algorithm coded *from scratch*. This can basically do the same as the "*from scratch*" algorithm in folder **SupportVectorMachine**, but this is much more complex to account for margins and more dimensions involved.
  * `regularSoftMarginSVM.py` - The algorithm coded with Scikit-learn (python library for machine learning).

### **_K-NearestNeighbors_** 
The K-nearest neighbors algorithm is a method used for classification and regression. The idea of K Nearest Neighbors classification is to best divide and separate the data based on clustering the data and to classify based on the proximity to it's K closest neighbors and their classifications. Where 'k' is the number of neighbors that are involved in the classification.

  * `breast-cancer-wisconsin.data` - Data used in the example.
  * `breast-cancer-wisconsin.names` - Information about the data used in the example.
  * `howItWorksKNearestNeighbors.py` - The algorithm coded *from scratch*.
  * `regularNearestNeighbors.py` - The algorithm coded with Scikit-learn (python library for machine learning).

### **_LogisticRegression_**
In statistics, logistic regression, or logit regression, or logit model is a regression model where the dependent variable (DV) is categorical. This project covers the case of a binary dependent variable—that is, where it can take only two values, "0" and "1", which represent outcomes such as pass/fail, win/lose, alive/dead or healthy/sick. 

The binary Logistic regression model is an example of a qualitative response/discrete choice model. It is used to estimate the probability of a binary response based on one or more predictor (or independent) variables (features).

### **_NaiveBayes_**
Currently empty, check back soon for updates.

### **_RandomForestClassification_**
Currently empty, check back soon for updates.

### **_SupportVectorMachine_** 
Support vector machines (SVMs) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. The objective of the SVM is to find the best separating hyperplane in vector space which is also referred to as the decision boundary.

The algorithm coded from scratch is a bare-bones and simple implementation to mainly show the thinking clearly without adding a lot of other elements. This algorithm can only take linearly separable data that does not overlap. For a more advanced implementation with overlapping and non-linear data see the folder **KernelSVM**.

  * `breast-cancer-wisconsin.data` - Data used in some of the examples.
  * `howItWorksSupportVectorMachine.py` - The algorithm coded *from scratch*.
  * `regularSupportVectorMachine.py` - The algorithm coded with Scikit-learn (python library for machine learning).
