# Deep Learning algorithms and projects
Python examples of Deep Learning. Each algorithm has it's own folder.

By making the algorithms from scratch I hope to show that I do not only know how to use the algorithms, but I also understand how it functions. By using conventional libraries for machine learning I hope to show that I can be effective and take advantage of the high-performance backend of libraries like for example, Scikit-learn and Tensorflow.
Folder names are **bolded** while filenames are `highlighted` for readability.

## FOLDER CONTENTS (in alphabetical order):

### **_ArtificialNeuralNetworks_**
Currently empty, check back soon for updates.

### **_ConvolutionalNeuralNetworks_**
Currently empty, check back soon for updates.

### **_DeepLearningWithTensorFlow_** 
Deep learning is part of a broader family of machine learning methods based on learning data representations, as opposed to task-specific algorithms. Learning can be supervised, partially supervised or unsupervised.

A deep neural network (DNN) is an artificial neural network with multiple hidden layers between the input and output layers. The algorithm is coded with Tensorflow. Tensorflow allows us to perform specific machine learning number-crunching operations on tensors with large efficiency.

All examples in this folder should be small enough to be reasonably assessed with the "only cpu" version of TensorFlow which was used when developing these examples.

  * `negative.txt` - file with negative sentiments from movie reviews.
  * `ownDataDeepLearningWithNeuralNetworks.py` - Deep learning with TensorFlow on either positive or negative sentiments with natural language processing (NLP) using the NLTK python library.
  * `create_sentiment_featuresets.py` - using natural language processing we create a featureset that is suitably preprocessed for machine learning in the `ownDataDeepLearningWithNeuralNetworks.py` file. **Warning:** *Will save a large pickle file if run directly, but will be called in the `ownDataDeepLearningWithNeuralNetworks.py` file*
  * `positive.txt` - file with positive sentiments from movie reviews.
  * `regularDeepLearningWithNeuralNetworks.py` - Deep learning with TensorFlow on data included in the tensorflow library.
