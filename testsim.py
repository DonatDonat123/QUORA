import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
import pickle
from nltk.corpus import wordnet as wn
import re
from common import *
import pickle
from scipy.sparse import lil_matrix, vstack
import numpy as np

gmax = 3.6375861597263857

def load_sparse_lil(filename):
    loader = np.load(filename)
    result = lil_matrix(tuple(loader["shape"]), dtype=str(loader["dtype"]))
    result.data = loader["data"]
    result.rows = loader["rows"]
    return result

def get_similarity(q1, q2, matrix, dictWords):
	sum = 0
	for w1 in q1:
		maximum = 0
		if w1 in dictWords:
			ind = dictWords[w1]
			for w2 in q2:
				if w2 in dictWords:
					jnd = dictWords[w2]
					if max(matrix[ind, jnd], matrix[jnd,ind]) > maximum:
							maximum = max(matrix[ind, jnd], matrix[jnd,ind])
							sum = sum + maximum
		else:
			if w1 in q2:
				sum = sum + gmax
	if len(q1) != 0:
		sim = sum / gmax / len(q1)
	else:
		sim = 0
	return sim, len(q1)-len(q2)
	
def get_similarities(q0, q1, matrix, dictNums, dictWords):
	similarities = []
	differences = []
	size = len(q0)
	for i in xrange(0,size):
		sim, diff = get_similarity(q0[i], q1[i], matrix, dictWords)
		similarities.append(sim)
		differences.append(diff)
	return similarities, differences

	
def make_submission(sim, diff, last_column, last_column_name, filename):
    sub = pd.DataFrame(data={"similarity":sim, "difference":diff, last_column_name:last_column})
    sub.to_csv(filename, columns=['similarity', "difference", last_column_name], index=False)

def main():
	print "Loading preprocessed/train100.bin        "
	traindata = loadObj('preprocessed/train100.bin')
	print "Train data loaded"
	
	print "Loading preprocessed/test.bin        "
	testdata = loadObj('preprocessed/test.bin')
	print "Test data loaded"
	
	print "Loading similarity matrices"
	simMatrix1 = load_sparse_lil('similarityMatrix1.npz')
	print simMatrix1.shape
	simMatrix2 = load_sparse_lil('similarityMatrix2.npz')
	print simMatrix2.shape
	print "Matrix loaded"
	
	M = vstack([simMatrix1,simMatrix2])
	matrix = lil_matrix(M)
	print matrix.shape
	
	print "Loading dictionaries"
	dictWords = loadObj('dictWords.bin')
	dictNums = loadObj('dictNums.bin')
	print "Dictionaries loaded"

	print "Computing similarities from train data"
	sim, diff = get_similarities(traindata[0], traindata[1], matrix, dictNums, dictWords)
	print "Done"
	
	print "Write data to file from train data"
	make_submission(sim, diff, traindata[2], "is_duplicate", "trainsubmission.csv")
	print "Done"
	
	print "Computing similarities from test data"
	sim, diff = get_similarities(testdata[0], testdata[1], "test_id", matrix, dictNums, dictWords)
	print "Done"
	
	print "Write data to file from test data"
	make_submission(sim, diff, testdata[2], "testsubmission.csv")
	print "Done"
	
	
if __name__ == "__main__":
    main()