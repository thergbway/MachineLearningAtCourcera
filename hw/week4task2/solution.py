import numpy
import pandas
from sklearn.decomposition import PCA

close_prices = pandas.read_csv('C:\\Users\\AND\\Desktop\\MachineLearningAtCourcera\\hw\\week4task2\\close_prices.csv')
pca = PCA(n_components=10)

del close_prices['date']

pca.fit(close_prices)
print("Components variance = " + str(pca.explained_variance_ratio_))

transformed_close_prices = pca.transform(close_prices)
first_component = pandas.DataFrame(transformed_close_prices)[0]

dj_index = pandas.read_csv('C:\\Users\\AND\\Desktop\\MachineLearningAtCourcera\\hw\\week4task2\\djia_index.csv')

corrcoef = numpy.corrcoef(pandas.DataFrame(data=[first_component, dj_index['^DJI']]))
print('corrcoef = ' + str(corrcoef))
