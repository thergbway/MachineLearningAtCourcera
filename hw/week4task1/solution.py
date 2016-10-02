import pandas
import scipy.sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

train_data = pandas.read_csv('C:\\Users\\AND\\Desktop\\MachineLearningAtCourcera\\hw\\week4task1\\salary-train.csv')
test_data = pandas.read_csv('C:\\Users\\AND\\Desktop\\MachineLearningAtCourcera\\hw\\week4task1\\salary-test-mini.csv')

train_data['FullDescription'] = [i for i in map(lambda text: text.lower(), train_data['FullDescription'])]

test_data['FullDescription'] = [i for i in map(lambda text: text.lower(), test_data['FullDescription'])]

train_data['FullDescription'] = train_data['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)

test_data['FullDescription'] = test_data['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)

vectorizer = TfidfVectorizer(min_df=5)
tfidfFullDescriptionTrainData = vectorizer.fit_transform(train_data['FullDescription'])
tfidfFullDescriptionTestData = vectorizer.transform(test_data['FullDescription'])

train_data['LocationNormalized'].fillna('nan', inplace=True)
train_data['ContractTime'].fillna('nan', inplace=True)

test_data['LocationNormalized'].fillna('nan', inplace=True)
test_data['ContractTime'].fillna('nan', inplace=True)

enc = DictVectorizer()
X_train_categ = enc.fit_transform(train_data[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(test_data[['LocationNormalized', 'ContractTime']].to_dict('records'))

train_data_completed = scipy.sparse.hstack([tfidfFullDescriptionTrainData, X_train_categ])
test_data_completed = scipy.sparse.hstack([tfidfFullDescriptionTestData, X_test_categ])

ridge = Ridge(alpha=1.0, random_state=241)
ridge.fit(train_data_completed, train_data['SalaryNormalized'])
predicted = ridge.predict(test_data_completed)

print("First prediction = " + str(predicted[0]))
print("Second prediction = " + str(predicted[1]))
