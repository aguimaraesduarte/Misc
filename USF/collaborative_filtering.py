import argparse
import re
import os
import csv
import math
import collections as coll
import numpy as np
#import time


def parse_argument():
	"""
	Code for parsing arguments
	"""
	parser = argparse.ArgumentParser(description='Parsing a file.')
	parser.add_argument('--train', nargs=1, required=True)
	parser.add_argument('--test', nargs=1, required=True)
	args = vars(parser.parse_args())

	return args


def parse_file(filename):
	"""
	Given a filename outputs user_ratings and movie_ratings dictionaries

	Input: filename
	Output: user_ratings, movie_ratings
		where:
			user_ratings[user_id] = {movie_id: rating}
			movie_ratings[movie_id] = {user_id: rating}
	"""
	user_ratings = {}
	movie_ratings = {}
	# Your code here

	with open(filename, "r") as infile:
		reader = csv.reader(infile)

		visited_movies = []
		visited_users = []

		for row in reader:
			movie = int(row[0])
			user = int(row[1])
			rating = float(row[2])

			if movie not in visited_movies:
				movie_ratings[movie] = {}
				visited_movies.append(movie)

			if user not in visited_users:
				user_ratings[user] = {}
				visited_users.append(user)

			movie_ratings[movie][user] = rating
			user_ratings[user][movie] = rating 

	return user_ratings, movie_ratings


def compute_average_user_ratings(user_ratings):
	""" Given a the user_rating dict compute average user ratings

	Input: user_ratings (dictionary of user, movies, ratings)
	Output: ave_ratings (dictionary of user and ave_ratings)
	"""
	ave_ratings = {}
	# Your code here

	for user in user_ratings:
		ave_ratings[user] = np.mean(user_ratings[user].values())

	return ave_ratings


def compute_user_similarity(d1, d2, ave_rat1, ave_rat2):
	""" Computes similarity between two users

		Input: d1, d2, (dictionary of user ratings per user) 
			ave_rat1, ave_rat2 average rating per user (float)
		Ouput: user similarity (float)
	"""
	# Your code here

	intersection = list(set(d1.keys()) & set(d2.keys()))

	if intersection == []:
		return 0.0
	else:
		numerator = 0
		denominator = 0
		denom1 = 0
		denom2 = 0

		for movie in intersection:
			numerator += (d1[movie] - ave_rat1) * (d2[movie] - ave_rat2)
			denom1 += (d1[movie] - ave_rat1)**2
			denom2 += (d2[movie] - ave_rat2)**2

		denominator = np.sqrt(denom1 * denom2)

		if numerator == 0 or denominator == 0:
			return 0.0
		else:
			return float(numerator)/denominator


def predict(i, k, user_ratings, movie_ratings, ave_ratings):
	ave_rati = ave_ratings[i]
	nominator = 0
	denominator = 0

	for j in movie_ratings[k]:
		ave_ratj = ave_ratings[j]
		w = compute_user_similarity(user_ratings[i], user_ratings[j], ave_rati, ave_ratj)
		nominator += w*(user_ratings[j][k] - ave_ratj)
		denominator += np.abs(w)

	if denominator == 0:
		return ave_rati
	else:
		return ave_rati + float(nominator)/denominator


def writeResults(predictions, movie_ratings):
	with open("predictions.txt", "w") as outfile:
		for movie in predictions:
			for user in predictions[movie]:
				outfile.write("{},{},{},{:.4f}".format(movie, user, movie_ratings[movie][user], predictions[movie][user])+os.linesep)


#def rmse(predictions, movie_ratings):
#	return np.sqrt(np.sum(np.sum([[(predictions[movie][user] - movie_ratings[movie][user])**2 for user in predictions[movie]] for movie in predictions]))/np.sum([len(a[1].keys()) for a in movie_ratings.items()]))


#def mae(predictions, movie_ratings):
#	return np.sum(np.sum([[np.abs(predictions[movie][user] - movie_ratings[movie][user]) for user in predictions[movie]] for movie in predictions]))/np.sum([len(a[1].keys()) for a in movie_ratings.items()])


def rmse(predictions, movie_ratings):
	sse = 0
	cnt = 0
	for movie in predictions:
		for user in predictions[movie]:
			sse += (predictions[movie][user] - movie_ratings[movie][user])**2
			cnt += 1
	mse = sse/cnt
	return np.sqrt(mse)

def mae(predictions, movie_ratings):
	ae = 0
	cnt = 0
	for movie in predictions:
		for user in predictions[movie]:
			ae += np.abs(predictions[movie][user] - movie_ratings[movie][user])
			cnt += 1
	mae = ae/cnt
	return mae

def main():
	"""
	This function is called from the command line via

	python cf.py --train [path to filename] --test [path to filename]
	"""
	args = parse_argument()
	train_file = args['train'][0]
	test_file = args['test'][0]
	#print train_file, test_file
	# your code here

	train_user_ratings, train_movie_ratings = parse_file(train_file)
	test_user_ratings, test_movie_ratings = parse_file(test_file)
	ave_user_ratings = compute_average_user_ratings(train_user_ratings)

	preds = {}
	for k in test_movie_ratings:
		preds[k] = {}

		for i in test_movie_ratings[k]:
			preds[k][i] = predict(i, k, train_user_ratings, train_movie_ratings, ave_user_ratings)
			
	writeResults(preds, test_movie_ratings)
	print "RMSE {:.4f}".format(rmse(preds, test_movie_ratings))
	print "MAE {:.4f}".format(mae(preds, test_movie_ratings))

if __name__ == '__main__':
	#t = time.time()
	main()
	#end = time.time() - t
	#print "{} seconds, {} minutes, {} hours".format(end, end/60., end/3600.)
