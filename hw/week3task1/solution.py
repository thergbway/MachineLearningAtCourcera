import numpy
import pandas

from sklearn.svm import SVC

from pandas.core.frame import DataFrame
from pandas.core.frame import Series

data = pandas.read_csv("C:\\Users\\AND\\Desktop\\MachineLearningAtCourcera\\hw\\week3task1\\svm-data.csv",
                       header=None)

target = data[0]
features = data[[1,2]]

svc = SVC(C=100000, kernel='linear', random_state=241)
svc.fit(features, target)

support = svc.support_

print('Indexes of the support elements = ' + str(support))