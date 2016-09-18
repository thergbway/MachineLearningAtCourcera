import pandas
import numpy as np

from pandas.core.frame import DataFrame

from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def load_data(path: str):
    return pandas.read_csv(path, header=None)


def get_features(data: DataFrame):
    return data[list(range(1, 14))]


def get_classes(data: DataFrame):
    return data[0]


def get_fold_generator(features: DataFrame):
    return KFold(n=len(features), n_folds=5, shuffle=True, random_state=42)


def get_k_neighbors_classifier(neighbors_count: int):
    return KNeighborsClassifier(n_neighbors=neighbors_count)


def get_avg_classifier_accuracy(features, classes, classifier, fold_generator):
    scores = cross_val_score(X=features, y=classes, estimator=classifier, cv=fold_generator)
    return np.average(scores)


def get_classifier_accuracy_test_results(features: DataFrame, classes: DataFrame):
    rst = DataFrame(columns=['neighbors', 'accuracy'])

    for i in range(1, 51):
        classifier = get_k_neighbors_classifier(i)
        foldGenerator = get_fold_generator(features)

        avg = get_avg_classifier_accuracy(
            features=features,
            classes=classes,
            classifier=classifier,
            fold_generator=foldGenerator
        )

        rst = rst.append(other={'neighbors': i, 'accuracy': avg}, ignore_index=True)

    return rst


def task_1_2():
    data = load_data("C:\\Users\\AND\\Desktop\\MachineLearningAtCourcera\\hw\\week2task1\\wine.data")
    features = get_features(data)
    classes = get_classes(data)

    rst = get_classifier_accuracy_test_results(features, classes)

    neighbors_best_count = int(rst.loc[lambda df: df.accuracy == rst.max().accuracy, :].neighbors)

    print("Without scaling: ")
    print("Best k=" + str(neighbors_best_count))
    print("Best accuracy=" + str(rst.max().accuracy))


def task_3_4():
    data = load_data("C:\\Users\\AND\\Desktop\\MachineLearningAtCourcera\\hw\\week2task1\\wine.data")
    features = get_features(data)
    classes = get_classes(data)

    # scaling
    features = DataFrame(scale(features), columns=range(1, 14))

    rst = get_classifier_accuracy_test_results(features, classes)

    neighbors_best_count = int(rst.loc[lambda df: df.accuracy == rst.max().accuracy, :].neighbors)
    print("With scaling: ")
    print("Best k=" + str(neighbors_best_count))
    print("Best accuracy=" + str(rst.max().accuracy))


task_1_2()
print()
task_3_4()
