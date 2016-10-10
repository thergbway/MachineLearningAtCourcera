import numpy as np
import pandas
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

data = pandas.read_csv('C:\\Users\\AND\\Desktop\\MachineLearningAtCourcera\\hw\\week5task2\\gbm-data.csv', header=None)

# data = data.values  # cast to numpy array

x_train, x_test, y_train, y_test = train_test_split(data[np.arange(1, 1777)].values, data[0].values, test_size=0.8,
                                                    random_state=241)

# for learning_rate in [1, 0.5, 0.3, 0.2, 0.1]:
#     csf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=learning_rate)
#     decision_fun_x_train = csf.staged_decision_function(x_train)
#     decision_fun_x_test = csf.staged_decision_function(x_test)

# csf = GradientBoostingClassifier(n_estimators=10, verbose=True, random_state=241, learning_rate=0.5)
# csf.fit(x_train, y_train)
# decision_fun_x_train = csf.staged_decision_function(x_train)
# decision_fun_x_test = csf.staged_decision_function(x_test)

# a = [x for x in map(lambda i, proba: i, enumerate(decision_fun_x_train))]

# [x for x in map(lambda elem: elem[1], top10)]

# x_train_array = pandas.DataFrame(columns=[0, 1])
# for i, proba in enumerate(decision_fun_x_train):
#     count = 0
#     for val in proba:
#         y_pred = 1 / (1 + np.exp(-val))
#         x_train_array.loc[count] = [i, y_pred]
#         count += 1

results = pandas.DataFrame(columns=[0, 1])

for n in np.arange(1, 251):
    csf = GradientBoostingClassifier(n_estimators=n, verbose=True, random_state=241, learning_rate=0.2)
    csf.fit(x_train, y_train)

    x_test_predict_proba = csf.predict_proba(x_test)
    test_loss = log_loss(y_test, x_test_predict_proba)
    # x_train_predict_proba = csf.predict_proba(x_train)
    # train_loss = log_loss(y_train, x_train_predict_proba)
    print('loss = ' + str(test_loss) + ', n = ' + str(n))
    results.loc[n] = [n, test_loss]

print(results)

classifier = RandomForestClassifier(n_estimators=37, random_state=241, verbose=True, n_jobs=4)
classifier.fit(x_train, y_train)

log_loss(y_test, classifier.predict_proba(x_test))
