import numpy
import pandas
import sklearn.datasets
import sklearn.preprocessing
import sklearn.cross_validation

from sklearn.neighbors import KNeighborsRegressor

from pandas.core.frame import DataFrame
from pandas.core.frame import Series

data = sklearn.datasets.load_boston()
features = DataFrame(data.data)
target = Series(data.target)

features = DataFrame(sklearn.preprocessing.scale(features))

linspace = Series(numpy.linspace(start=1, stop=10, num=200))

foldGenerator = sklearn.cross_validation.KFold(len(features), 5, shuffle=True, random_state=42)

results = DataFrame(columns=['p', 'avg_mean_sq_error'])

for p in linspace:
    regressor = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=p)
    scores = sklearn.cross_validation.cross_val_score(estimator=regressor, X=features, y=target,
                                                      scoring='mean_squared_error', cv=foldGenerator)

    avg_score = numpy.average(scores)
    avg_score *= -1  # just to make things clear
    results = results.append(other={'p': p, 'avg_mean_sq_error': avg_score}, ignore_index=True)

print("Results: ")
print(results)
print()

best_p = int(results.loc[lambda df: df.avg_mean_sq_error == results.min().avg_mean_sq_error, :].p)

print("Best p = " + str(best_p))
print("Min avg_mean_sq_error = " + str(results.min().avg_mean_sq_error))
