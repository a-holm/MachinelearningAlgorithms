import numpy as np

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# tpot_data = np.recfromcsv('Arundo_take_home_challenge_training_set.csv', delimiter=',', dtype=np.float64)
# features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('request_count'), axis=1)
# training_features, testing_features, training_target, testing_target = \
#     train_test_split(features, tpot_data['request_count'], random_state=42)
dataset = pd.read_csv('Arundo_take_home_challenge_training_set.csv')
# print(dataset.head())
features = dataset.iloc[:, [1, 3, 4, 5, 6, 7]].values
labels = dataset.iloc[:, 2].values

# Encode categorical data and make them into numbers
labelEncoder_x = LabelEncoder()
features[:, 5] = labelEncoder_x.fit_transform(features[:, 5])
# make dummy columns to avoid attributing order
# onehotencoder = OneHotEncoder(categorical_features=[5])
# features = onehotencoder.fit_transform(features).toarray()

# # Avoiding the Dummy variable trap
# features = features[:, 1:]

feature_train, feature_test, label_train, label_test = train_test_split(
        features, labels, test_size=0.2)

exported_pipeline = make_pipeline(
    SelectFromModel(estimator=ExtraTreesRegressor(max_features=0.5, n_estimators=100), threshold=0.1),
    LassoLarsCV(normalize=True)
)

exported_pipeline.fit(feature_train, label_train)
results = exported_pipeline.predict(feature_test)
