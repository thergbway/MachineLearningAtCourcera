import pandas
import sklearn.metrics

classification_data = pandas.read_csv(
    'C:\\Users\\AND\\Desktop\\MachineLearningAtCourcera\\hw\\week3task4\\classification.csv')

clf_true = classification_data['true']
clf_pred = classification_data['pred']

# 1
TP = sum([i for i in map(lambda true, pred: pred == 1 and true == 1, clf_true, clf_pred)])
FP = sum([i for i in map(lambda true, pred: pred == 1 and true == 0, clf_true, clf_pred)])
FN = sum([i for i in map(lambda true, pred: pred == 0 and true == 1, clf_true, clf_pred)])
TN = sum([i for i in map(lambda true, pred: pred == 0 and true == 0, clf_true, clf_pred)])

print('TP=' + str(TP))
print('FP=' + str(FP))
print('FN=' + str(FN))
print('TN=' + str(TN))
print()
print()

# 2
accuracy = sklearn.metrics.accuracy_score(clf_true, clf_pred)
precision = sklearn.metrics.precision_score(clf_true, clf_pred)
recall = sklearn.metrics.recall_score(clf_true, clf_pred)
f1_score = sklearn.metrics.f1_score(clf_true, clf_pred)

print("Accuracy = " + str(accuracy))
print("Precision = " + str(precision))
print("Recall = " + str(recall))
print("F1 score = " + str(f1_score))
print()
print()

# 3
scores = pandas.read_csv('C:\\Users\\AND\\Desktop\\MachineLearningAtCourcera\\hw\\week3task4\\scores.csv')
score_logreg = sklearn.metrics.roc_auc_score(scores['true'], scores['score_logreg'])
score_svm = sklearn.metrics.roc_auc_score(scores['true'], scores['score_svm'])
score_knn = sklearn.metrics.roc_auc_score(scores['true'], scores['score_knn'])
score_tree = sklearn.metrics.roc_auc_score(scores['true'], scores['score_tree'])

print('Score logreg = ' + str(score_logreg))
print('Score svm = ' + str(score_svm))
print('Score knn = ' + str(score_knn))
print('Score tree = ' + str(score_tree))
print()
print()


# 4
def max_accuracy(y_true, y_pred):
    pres_recall_threshold_typle = sklearn.metrics.precision_recall_curve(y_true, y_pred)
    c = pandas.DataFrame(data={0: pres_recall_threshold_typle[0], 1: pres_recall_threshold_typle[1]})
    return max(c.loc[lambda df: df[1] > 0.7][0])


max_acc_for_logreg = max_accuracy(scores['true'], scores['score_logreg'])
max_acc_for_svm = max_accuracy(scores['true'], scores['score_svm'])
max_acc_for_knn = max_accuracy(scores['true'], scores['score_knn'])
max_acc_for_tree = max_accuracy(scores['true'], scores['score_tree'])

print('Max accuracy for logreg = ' + str(max_acc_for_logreg))
print('Max accuracy for svm = ' + str(max_acc_for_svm))
print('Max accuracy for knn = ' + str(max_acc_for_knn))
print('Max accuracy for tree = ' + str(max_acc_for_tree))
