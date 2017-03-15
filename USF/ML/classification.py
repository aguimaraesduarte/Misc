#execfile('separate.py')

# Prints some "how to use this script" instructions
def help ():
	print 'Arguments by position:'
	print '\t1: Train file    - Full or relative path'
	print '\t2: Test file     - Full or relative path'
	print '\t3: Algorithm     - One of {LDA|QDA|kNN|logit}'
	print '\t4: Optional      - Number of neighbors for kNN (default=5)'


# Populates and returns a dictionary with options from command line
def get_args_by_position (args):
	opts = {}
	if len(args) != 4 and len(args) != 5:
		return opts
	opts['train_file'] = args[1]
	opts['test_file'] = args[2]

	if args[3].lower() == 'lda':
		opts['algo'] = 'LDA'
	elif args[3].lower() == 'qda':
		opts['algo'] = 'QDA'
	elif args[3].lower() == 'knn':
		opts['algo'] = 'KNN'
	elif args[3].lower() == 'logit':
		opts['algo'] = 'LOGIT'
	else:
		sys.exit("Your algorithm '%s' is not one of {LDA|QDA|kNN|logit}" % args[3].lower())
	if len(args) == 5:
		opts['algo_arg'] = args[4]
	return opts


# Reads a CSV file
# Returns the first column(s) as targets; last column as data
# Consider replacing this with pandas
def read_csv (file_name):
	import csv
	data = []
	targets = []
	with open(file_name, 'rb') as csvfile:
		lines = csv.reader(csvfile)
		for row in lines:
			targets.append(row[0])
			data.append(", ".join(row[1:]))
	return data, targets


# Calculates and returns accuracy
def calculate_accuracy (targets, hypotheses):
	from sklearn import metrics
	return metrics.accuracy_score(targets, hypotheses)


# Returns the algorithm according to what's in opts
def get_algo (opts):
	from sklearn.linear_model import LogisticRegression
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
	from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
	#
	if opts['algo'].lower() == 'lda':
		return LinearDiscriminantAnalysis()
	elif opts['algo'].lower() == 'qda':
		return QuadraticDiscriminantAnalysis()
	elif opts['algo'].lower() == 'knn':
		neighbours = 5     # Default to k=5
		# ... but take options from command line, if it's there.
		if 'algo_arg' in opts:
			neighbours = int(opts['algo_arg'])
		return KNeighborsClassifier(n_neighbors=neighbours)
	elif opts['algo'].lower() == 'logit':
		return LogisticRegression()

# Separate a tweet into tokens
def tokenize(tweet):
	import re
	return re.findall("'\w+|[\w]+|[^\s\w]", tweet)[1:-1]


# return the count of a token from a list of tokens
def count_occurrences(token, list_tokens):
	return list_tokens.count(token)

# return features as list of lists
def get_counts(list_tokens, most_common):
	rates = []
	n = float(len(list_tokens))
	# rates of occurrences from function_words
	for word in function_words:
		rates.append(count_occurrences(word, list_tokens)/n)
	# rates of occurrences from punctuation_symbols
	for punctuation in punctuation_symbols:
		rates.append(count_occurrences(punctuation, list_tokens)/n)
	# rates of occurrences from most common words
	for token in most_common:
		rates.append(count_occurrences(token, list_tokens)/n)
	# number of tokens
	rates.append(n)
	# rates of positive words
	for word in positive_words:
		rates.append(count_occurrences(word, list_tokens)/n)
	# rates of negative words
	for word in negative_words:
		rates.append(count_occurrences(word, list_tokens)/n)
	return rates

# find 1000 most common words in tweets
def get_common(list_tweets, n=1000):
	from collections import Counter
	tokens = [token for tweet in list_tweets for token in tweet]
	counter = Counter(tokens)
	most_common = [tup[0] for tup in counter.most_common(n)]
	return list(set(most_common)-set(function_words)-set(punctuation_symbols))


# Drives the script by acting as a manager
def process_data (opts):
	from sklearn.cross_validation import cross_val_score
	#	
	train_tweets, train_sentiments = read_csv(opts['train_file'])
	test_tweets, test_sentiments = read_csv(opts['test_file'])
	#
	train_tokenized_tweets = map(tokenize, train_tweets)
	#
	most_common = get_common(train_tokenized_tweets)
	train_rates = [get_counts(tokenized_tweet, most_common) for tokenized_tweet in train_tokenized_tweets]
	#
	test_tokenized_tweets = map(tokenize, test_tweets)
	test_rates = [get_counts(tokenized_tweet, most_common) for tokenized_tweet in test_tokenized_tweets]
	#
	algo = get_algo(opts)
	if algo is not None:
        	algo.fit(train_rates, train_sentiments)
		hypotheses = algo.predict(test_rates)
		print 'Misclassification Rate = %.4f' % (1-calculate_accuracy(test_sentiments, hypotheses))
	else:
		print 'Cannot generate predictions or accuracy. (Are there options on command line?)'

	calc_CV = False
	if calc_CV:
		scores = cross_val_score(algo, train_rates, train_sentiments, cv=10)
		print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

# main

import sys
opts = get_args_by_position(sys.argv)
if len(opts) == 0:
	help()
else:
	function_words = ["I", "the", "and", "to", "a", "of", "that", "in", "it", "my", "is",
                      "you", "was", "for", "have", "with", "he", "me", "on", "but"]
	punctuation_symbols = [".", ",", "!"]
	positive_words = ["amazing", "awesome", "incredible", "unforgettable", "great",
	                  "fantastic", "good", "happy", "best"]
	negative_words = ["terrible", "horrible", "sad", "worst", "horrid"]
	process_data(opts)






