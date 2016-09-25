import math

import pandas
from pandas.core.frame import Series
from sklearn.metrics import roc_auc_score


def new_w(prev_w: float, another_prev_w: float, features: Series, another_features: Series, target: Series, k: float,
          C: float):
    def f(i):
        return target[i] * features[i] * (
            1.0 - 1 / (1 + math.exp(-target[i] * (prev_w * features[i] + another_prev_w * another_features[i]))))

    inner_sum = sum([elem for elem in map(f, range(len(target)))])

    return prev_w + k / len(target) * inner_sum - k * C * prev_w


def fit(features1: Series, features2: Series, target: Series, k=0.1, C=0.0, max_dist=1e-5):
    w0 = 0.0
    w1 = 0.0
    prev_dist = max_dist + 100.0

    iter_num = 0
    while prev_dist > max_dist and iter_num <= 10000:
        if(iter_num%10 == 0):
            print("iter_num=" + str(iter_num) + ", dist=" + str(prev_dist))
        iter_num += 1
        prev_w0 = w0
        prev_w1 = w1
        w0 = new_w(w0, w1, features1, features2, target, k, C)
        w1 = new_w(w1, w0, features2, features1, target, k, C)
        prev_dist = math.sqrt((w0 - prev_w0) * (w0 - prev_w0) + (w1 - prev_w1) * (w1 - prev_w1))

    print("total iterations=" + str(iter_num) + ", final dist=" + str(prev_dist))
    return {'w0': w0, 'w1': w1}


def get_a(w0: float, w1: float, x1: float, x2: float):
    return 1 / (1 + math.exp(-w0 * x1 - w1 * x2))


def get_roc_auc_score(target: Series, w0, w1, features1: Series, features2: Series):
    probability_series = [i for i in map(lambda x0, x1: get_a(w0, w1, x0, x1), features1, features2)]

    return roc_auc_score(target, probability_series)


data = pandas.read_csv('C:\\Users\\AND\\Desktop\\MachineLearningAtCourcera\\hw\\week3task3\\data-logistic.csv',
                       header=None)

target = data[0]
features1 = data[1]
features2 = data[2]

# no regularization
res1 = fit(features1, features2, target)
score1 = get_roc_auc_score(target, res1['w0'], res1['w1'], features1, features2)
print("score with no regularization=" + str(score1))

# with regularization
res2 = fit(features1, features2, target, C=10.0)
score2 = get_roc_auc_score(target, res2['w0'], res2['w1'], features1, features2)
print("score with regularization=" + str(score2))