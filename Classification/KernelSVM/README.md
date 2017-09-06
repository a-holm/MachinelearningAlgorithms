### **_KernelSVM_** 
Kernel SVM is basically an SVM (see folder **SupportVectorMachine**) which operates with kernels. Kernels use the inner product to help us transform the feature space to make it possible for Support Vector Machines to create a good hyperplane  to effectively classify non-linear feature sets.

When I say that the SVM has a "soft margin" then that means that it has some 'slack' and allows features to be 'wrongly' classified to avoid overfitting the classifier. This also includes kernels. 

  * `breast-cancer-wisconsin.data` - Data used in some of the examples.
  * `howItWorksSoftMarginSVM.py` - The algorithm coded *from scratch*. This can basically do the same as the "*from scratch*" algorithm in folder **SupportVectorMachine**, but this is much more complex to account for margins and more dimensions involved.
  * `regularSoftMarginSVM.py` - The algorithm coded with Scikit-learn (python library for machine learning).
  * `regularKernelSVM.py`- Kernel SVM in Python
  * `regularKernelSVM.r`- Kernel SVM in R
  * `Social_Network_Ads.csv` - The Data used in `regularKernelSVM.py` and `regularKernelSVM.r`.