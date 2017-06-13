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

	
def make_submission(sim, diff, ids):
    sub = pd.DataFrame(data={"test_id":ids,"similarity":sim, "difference":diff})
    sub.to_csv('sub.csv', columns=['test_id', 'similarity', "difference"], index=False)

def main():
	print "Loading preprocessed/test.bin        "
	data = loadObj('preprocessed/test.bin')
	print "Data loaded"
	print len(data[2])
	
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

	print "Computing similarities"
	sim, diff = get_similarities(data[0], data[1], matrix, dictNums, dictWords)
	print sim
	print "Done"
	
	ids = range(0, len(data[0]))
	print len(sim)
	print "Making submission"
	make_submission(sim, diff, ids)
	print "Done"
	
if __name__ == "__main__":
    main()