import decimal
import pandas
import numpy
import sklearn.metrics
import sklearn.preprocessing

from sklearn.linear_model import Perceptron

train_data = pandas.read_csv("C:\\Users\\AND\\Desktop\\MachineLearningAtCourcera\\hw\\week2task3\\perceptron_train.csv",
                             header=None)
test_data = pandas.read_csv("C:\\Users\\AND\\Desktop\\MachineLearningAtCourcera\\hw\\week2task3\\perceptron_test.csv",
                            header=None)

train_target = train_data[0]
train_features = train_data[[1, 2]]

test_target = test_data[0]
test_features = test_data[[1, 2]]

clf = Perceptron(random_state=241)
clf.fit(train_features, train_target)

predicted_target = clf.predict(test_features)

accuracy_score = sklearn.metrics.accuracy_score(test_target, predicted_target)

print("Accuracy score WITHOUT SCALING = " + str(accuracy_score))

std_scaler = sklearn.preprocessing.StandardScaler()
train_features_scaled = std_scaler.fit_transform(train_features)
test_features_scaled = std_scaler.transform(test_features)

clf = Perceptron(random_state=241)
clf.fit(train_features_scaled, train_target)

predicted_target_scaled = clf.predict(test_features_scaled)
accuracy_score_with_scaled_features = sklearn.metrics.accuracy_score(test_target, predicted_target_scaled)

print("Accuracy score WITH SCALING = " + str(accuracy_score_with_scaled_features))

diff = accuracy_score_with_scaled_features - accuracy_score
print("Difference between accuracies = {diff:.3f}".format(diff=diff))
