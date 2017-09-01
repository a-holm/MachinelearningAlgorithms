# Clustering algorithms and projects
Python examples of Clustering. Each algorithm has it's own folder.

By making the algorithms from scratch I hope to show that I do not only know how to use the algorithms, but I also understand how it functions. By using conventional libraries for machine learning I hope to show that I can be effective and take advantage of the high-performance backend of libraries like for example, Scikit-learn and Tensorflow.

Folder names are **bolded** while filenames are `highlighted` for readability.

## FOLDER CONTENTS (in alphabetical order):

### **_HierarchicalClustering_**
Currently empty, check back soon for updates.

### **_K-MeansClustering_** 
K-means clustering is a unsupervised method to cluser or group the data. K-means allows you to choose the number (k) of categories/groups and categorizes it automatically when it has come up with solid categories.

This algorithm and other unsupervised algorithms is usually used to research the data and finding structure so it is not expected to be super precise.

  * `howItWorksKMeans.py` - The algorithm coded *from scratch*.
  * `regularKMeans.py` - The algorithm coded with Scikit-learn (python library for machine learning).
  * `titanicKMeans.py` - This is similar to `regularKMeans.py`, but is more advanced and uses an imported titanic.xls file which contains non-numeric data so that I can how I would handle such data.
  * `titanic.xls` - Data that looks like a passager list on the Titanic.

### **_MeanShiftClustering_** 
Mean Shift is very similar to the K-Means algorithm (see folder **kMeans**), except for one very important factor: you do not need to specify the number of groups prior to training. The Mean Shift algorithm finds clusters on its own. For this reason, it is even more of an "unsupervised" machine learning algorithm than K-Means.

  * `howItWorksMeanShift.py` - The algorithm coded *from scratch*.
  * `regularMeanShift.py` - The algorithm coded with Scikit-learn (python library for machine learning).
  * `titanic.xls` - Data that looks like a passager list on the Titanic.
  * `titanicKMeans.py` - This is similar to `regularMeanShift.py`, but is more advanced and uses an imported titanic.xls file which contains non-numeric data so that I can how I would handle such data.
