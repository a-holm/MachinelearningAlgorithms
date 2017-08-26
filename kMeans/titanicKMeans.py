# -*- coding: utf-8 -*-
"""K-Means unsupervised classification for machine learning.

K-means clustering is a unsupervised method to cluser or group the data.
K-means allows you to choose the number (k) of categories/groups and
categorizes it automatically when it has come up with solid categories.

This algorithm is usually used for research and finding structure and is not
expected to be super precise.

This file uses an imported titanic.xls file which contains non-numeric data and
shows how I would deal with such data. The data is found on the internet, but
the original source is unknown.

Example:

        $ python titanicKMeans.py

Todo:
    *
"""
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
style.use('ggplot')

"""
The data in the titanic.xls file looks like this:
------------------------------------------------
pclass - Passenger Class (1=1st; 2=2nd; 3=3rd)
survival - Survival (0=No; 1=Yes)
name - Name
sex - Sex
age - Age
sibsp - Number of Siblings/Spouses Aboard
parch - Number of Parents/Children Aboard
ticket - Ticket Number
fare - Passenger Fare (British pound)
cabin - Cabin
embarked - Port of Embarkation (C=Cherbourg; Q=Queenstown; S=Southampton)
boat - Lifeboat
body - Body Identification Number
home.dest - Home/Destination
"""
