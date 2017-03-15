import numpy as np
import argparse
from sklearn.tree import DecisionTreeClassifier

def parse_argument():
	"""
	Code for parsing arguments
	"""
	parser = argparse.ArgumentParser(description='Parsing a file.')
	parser.add_argument('--train', nargs=1, required=True)
	parser.add_argument('--test', nargs=1, required=True)
	parser.add_argument('--numTrees', nargs=1, required=True)
	args = vars(parser.parse_args())
	return args


def adaboost(X, y, num_iter):
	"""Given an numpy matrix X, a array y and num_iter return trees and weights 
	Input: X, y, num_iter
	Outputs: array of trees from DecisionTreeClassifier
			 trees_weights array of floats
	Assumes y is in {-1, 1}^n
	"""
	trees = []
	trees_weights = []
	# your code here
	N = X.shape[0]
	# initialize weigths
	w = np.array([1.0/N for i in range(N)])
	for t in range(num_iter):
		b = DecisionTreeClassifier(max_depth=1)
		b.fit(X, y, sample_weight=w)
		b_pred = b.predict(X)
		error = b_pred!=y
		error = np.mean(np.average(error, weights=w, axis=0))
		# Stop if classification is perfect
		if error <= 0:
			trees.append(b)
			trees_weights.append(1)
			return trees, trees_weights
		alpha = np.log((1.-error)/error)
		for i in range(N):
			if b_pred[i]!=y[i]:
				w[i] *= np.exp(alpha)
		trees.append(b)
		trees_weights.append(alpha)
	return trees, trees_weights


def adaboost_predict(X, trees, trees_weights):
	"""Given X, trees and weights predict Y
	assume Y in {-1, 1}^n
	"""
	# your code here
	Yhat = np.zeros(X.shape[0])

	for i in range(len(trees)):
		Yhat += trees_weights[i]*trees[i].predict(X)

	Yhat = [np.sign(y) for y in Yhat]

	return Yhat


def parse_spambase_data(filename):
	""" Given a filename return X and Y numpy arrays
	X is of size number of rows x num_features
	Y is an array of size the number of rows
	Y is the last element of each row.
	"""
	# your code here
	df = pd.read_csv(filename, sep=",", header=None)
	X = df.drop(df.shape[1]-1, axis=1).values
	Y = df.iloc[:,-1].values
	return X, Y


def new_label(Y):
	""" Transforms a vector od 0s and 1s in -1s and 1s.
	"""
	return [-1. if y == 0. else 1. for y in Y]


def old_label(Y):
	return [0. if y == -1. else 1. for y in Y]


def accuracy(y, pred):
	return np.sum([y[i]==pred[i] for i in range(len(y))])/float(len(y))
	#return np.sum(y == pred) / float(len(y)) 


def writeResults(x, y, pred):
	with open("predictions.txt", "w") as outfile:
		for i in range(len(y)):
			for xi in x[i]:
				outfile.write("{},".format(xi))
			outfile.write("{},{}".format(int(y[i]), int(pred[i]))+os.linesep)


def main():
	"""
	This code is called from the command line via
	python adaboost.py --train [path to filename] --test [path to filename] --numTrees 
	"""
	args = parse_argument()
	train_file = args['train'][0]
	test_file = args['test'][0]
	num_trees = int(args['numTrees'][0])
	#print train_file, test_file, num_trees
	# your code here
	# read data as numpy arrays
	X_train, Y_train = parse_spambase_data(train_file)
	X_test, Y_test = parse_spambase_data(test_file)
	Y_train = new_label(Y_train)
	Y_test = new_label(Y_test)
	t, tw = adaboost(X_train, Y_train, num_trees)
	Yhat = adaboost_predict(X_train, t, tw)
	Yhat_test = adaboost_predict(X_test, t, tw)
	## here print accuracy and write predictions to a file
	writeResults(X_test, old_label(Y_test), old_label(Yhat_test))

	# print 
	# print "\tMy implementation of Adaboost"
	acc_test = accuracy(Y_test, Yhat_test)
	acc = accuracy(Y_train, Yhat)
	print("Train Accuracy %.4f" % acc)
	print("Test Accuracy %.4f" % acc_test)


if __name__ == '__main__':
	main()

# accuracy =~ 0.91 with 10 trees

