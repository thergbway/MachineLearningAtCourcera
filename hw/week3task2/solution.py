import numpy as np
from sklearn import datasets
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

newsgroups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space']
)
features = newsgroups.data
target = newsgroups.target

vectorizer = TfidfVectorizer()
tfIdfFeatures = vectorizer.fit_transform(features)
features_mapping = vectorizer.get_feature_names()

grid = {'C': np.power(10.0, np.arange(-5, +5))}
cv = KFold(len(features), n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(tfIdfFeatures, target)

scores_ = gs.grid_scores_
print('Got scores: ')
for a in scores_:
    print(a)

best_C_param = gs.best_params_['C'];
print("\nBest C = " + str(best_C_param))

full_clf = SVC(C=1.0, kernel='linear', random_state=241)
full_clf.fit(tfIdfFeatures, target)

coef_ = np.abs(full_clf.coef_.toarray()[0])
top10 = sorted(zip(coef_, features_mapping))[-10:]

top10.sort(key=lambda a: a[1])

top10SortedWords = [x for x in map(lambda elem: elem[1], top10)]
print("Result:")
for i in range(0, len(top10SortedWords) - 1):
    print(top10SortedWords[i], end=",", flush=True)

print(top10SortedWords[len(top10SortedWords) - 1], end="", flush=True)
