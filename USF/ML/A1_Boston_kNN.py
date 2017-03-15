from sklearn.datasets import load_boston
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor

# load data
boston = load_boston()

data = boston.data # 506 lines of 13 features
feature_names = boston.feature_names # 13 features
target = boston.target # 506 target

# split into train and test sets
train_x = data[:-50]
train_y = target[:-50]
test_x = data[-50:]
test_y = target[-50:]

k = 3 # Can be changed to any integer > 0
algo = KNeighborsRegressor(n_neighbors=k)
algo.fit(train_x, train_y)
hypotheses = algo.predict(test_x)
print "MSE: %.3f (k = %d)" % (metrics.mean_squared_error(test_y, hypotheses), k)