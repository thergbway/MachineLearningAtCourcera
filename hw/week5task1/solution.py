import numpy as np
import pandas
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV

data = pandas.read_csv('C:\\Users\\AND\\Desktop\\MachineLearningAtCourcera\\hw\\week5task1\\abalone.csv')

data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

features = data[
    ['Sex', 'Length', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight', 'ShellWeight', ]
]

target = data['Rings']

grid = {'n_estimators': np.arange(1, 51)}
k_fold = KFold(n=len(features), n_folds=5, shuffle=True, random_state=1)
regressor = RandomForestRegressor(random_state=1)
gs = GridSearchCV(regressor, grid, scoring='r2', cv=k_fold)
gs.fit(features, target)

scores_ = gs.grid_scores_
print('Got scores. Find your answer here: ')
for a in scores_:
    print(a)
