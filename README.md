# Machine learning and Artificial intelligence projects in Python and R
###### Technologies involved: Python, R, Matplotlib, Numpy, SciPY, Skikit-learn, TensorFlow

Examples of machine learning algorithms and applied machine learning. Each algorithm has it's own folder and the plan is to eventually fill this repository of examples from every machine learning algorithm mentioned on Wikipedia.

By making the algorithms from scratch I hope to show that I do not only know how to use the algorithms, but I also understand how it functions. By using conventional libraries for machine learning I hope to show that I can be effective and take advantage of the high-performance backend of libraries like for example, Scikit-learn and Tensorflow.

I will refer to independent variables (also just called X) as 'features' and I will refer to the dependent variables (also just called y) as 'labels' due to convention.

##### EACH FOLDER HAS IT'S OWN `README.md` FILE WITH MORE INFORMATION

Folder names are **bolded** while filenames and warnings are `highlighted` for readability.

## FOLDER CONTENTS (in alphabetical order):

### **_Association Rule Learning_** 
Association rules are if/then statements that help uncover relationships between seemingly unrelated data in a relational database or other information repository. An example of an association rule would be "If a customer buys a dozen eggs, he is 80% likely to also purchase milk."

  - **Apriori**:  `(empty, will be updated soon)`
  - **Eclat**:  `(empty, will be updated soon)`

### **_[Classification](https://github.com/a-holm/MachinelearningAlgorithms/tree/master/Classification)_** 
Classification is the problem of identifying to which of a set of categories (sub-populations) a new observation belongs, on the basis of a training set of data containing observations (or instances) whose category membership is known. An example would be assigning a given email into "spam" or "non-spam" classes or assigning a diagnosis to a given patient as described by observed characteristics of the patient (gender, blood pressure, presence or absence of certain symptoms, etc.).

  - **DecisionTreeClassifiaction**:  `(empty, will be updated soon)`
  - **[KernelSVM](https://github.com/a-holm/MachinelearningAlgorithms/tree/master/Classification/KernelSVM)**: 
     Examples of the Kernel SVM algorithm. **`(Includes algorithm from scratch)`**
  - **[K-NearestNeighbors](https://github.com/a-holm/MachinelearningAlgorithms/tree/master/Classification/K-NearestNeighbors)**: 
     Examples of the K nearest Neighbors algorithm. **`(Includes algorithm from scratch)`**
  - **LogisticRegression**:  `(empty, will be updated soon)`
  - **NaiveBayes**:  `(empty, will be updated soon)`
  - **RandomForestClassifiaction**:  `(empty, will be updated soon)`
  - **[SupportVectorMachine](https://github.com/a-holm/MachinelearningAlgorithms/tree/master/Classification/SupportVectorMachine)**: 
     Examples of general Support Vector Machines. **`(Includes algorithm from scratch)`**

### **_[Clustering](https://github.com/a-holm/MachinelearningAlgorithms/tree/master/Clustering)_** 
Cluster analysis or clustering is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense or another) to each other than to those in other groups (clusters). These algorithms are often used to research and explore the data to make more focused categorization later.

  - **HierarchicalClustering**:  `(empty, will be updated soon)`
  - **[K-MeansClustering](https://github.com/a-holm/MachinelearningAlgorithms/tree/master/Clustering/K-MeansClustering)**:
     Examples of K-Means Clustering. **`(Includes algorithm from scratch)`**
  - **[MeanShiftClustering](https://github.com/a-holm/MachinelearningAlgorithms/tree/master/Clustering/MeanShiftClustering)**:
     Examples of Mean Shift Clustering. **`(Includes algorithm from scratch)`**

### **_[DataPreprocessing](https://github.com/a-holm/MachinelearningAlgorithms/tree/master/DataPreprocessing)_** 
Contains a Python and R template for pre-processing machine learning data.

  - **data**: Just sample data for testing.
  - `dataPreprocessingTemplate.py`: Python template for preprocessing.
  - `dataPreprocessingTemplate.r`: R template for preprocessing.

### **_[Deep Learning](https://github.com/a-holm/MachinelearningAlgorithms/tree/master/Deep%20Learning)_** 
Deep learning is the fastest-growing field in machine learning. It uses many-layered Deep Neural Networks (DNNs) to learn levels of representation and abstraction that make sense of data such as images, sound, and text.

  - **ArtificialNeuralNetworks**:  `(empty, will be updated soon)`
  - **ConvolutionalNeuralNetworks**: `(empty, will be updated soon)`
  - **[DeepLearningWithTensorFlow](https://github.com/a-holm/MachinelearningAlgorithms/tree/master/Deep%20Learning/DeepLearningWithTensorFlow)**:
     Deep learning project with TensorFlow.

### **_Dimensionality Reduction_** 
In machine learning and statistics, dimensionality reduction or dimension reduction is the process of reducing the number of random variables under consideration.

  - **KernelPCA**:  `(empty, will be updated soon)`
  - **LinearDiscriminantAnalysis-LDA**: `(empty, will be updated soon)`
  - **PrincipalComponentAnalysis-PCA**: `(empty, will be updated soon)`

### **_Model Selection Boosting_** 
This folder include Model Selection and Boosting. Model selection is the task of selecting a statistical model from a set of candidate models, given data. In the simplest cases, a pre-existing set of data is considered. Boosting is a machine learning meta-algorithm for primarily reducing bias and variance.

  - **GradientBoostingWithXGBoost**:  `(empty, will be updated soon)`
  - **ModelSelection**: `(empty, will be updated soon)`

### **_[Regression](https://github.com/a-holm/MachinelearningAlgorithms/tree/master/Regression)_** 
regression analysis is a set of statistical processes for estimating the relationships among variables.

  - **DecisionTreeRegression**:  `(empty, will be updated soon)`
  - **[MultipleLinearRegression](https://github.com/a-holm/MachinelearningAlgorithms/tree/master/Regression/MultipleLinearRegression)**:  Examples of Multiple Linear Regression. 
  - **[PolynomialRegression](https://github.com/a-holm/MachinelearningAlgorithms/tree/master/Regression/PolynomialRegression)**:
  Examples of Polynomial Regression. 
  - **RandomForestRegression**:  `(empty, will be updated soon)`
  - **[SimpleLinearRegression](https://github.com/a-holm/MachinelearningAlgorithms/tree/master/Regression/SimpleLinearRegression)**: 
    Examples of Simple Linear Regression. **`(Includes algorithm from scratch)`**
  - **SupportVectorRegression**:  `(empty, will be updated soon)`

### **_Reinforcement Learning_** 
Reinforcement learning (RL) is an area of machine learning inspired by behaviorist psychology, concerned with how software agents ought to take actions in an environment so as to maximize some notion of cumulative reward.

  - **ThompsonSampling**:  `(empty, will be updated soon)`
  - **UpperConfidenceBound**: `(empty, will be updated soon)`