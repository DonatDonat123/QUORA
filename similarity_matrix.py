from common import *
from scipy.sparse import lil_matrix
from nltk.corpus import wordnet as wn
import io

import numpy as np

def save_sparse_lil(filename, array):
    # use np.savez_compressed(..) for compression
    np.savez(filename, dtype=array.dtype.str, data=array.data,
        rows=array.rows, shape=array.shape)

def save_sparse_lil_compressed(filename, array):
    np.savez_compressed(filename, dtype=array.dtype.str, data=array.data,
        rows=array.rows, shape=array.shape)

def load_sparse_lil(filename):
    loader = np.load(filename)
    result = lil_matrix(tuple(loader["shape"]), dtype=str(loader["dtype"]))
    result.data = loader["data"]
    result.rows = loader["rows"]
    return result
	
def main():
	dictWords = loadObj('dictWords.bin')
	dictNums = loadObj('dictNums.bin')
	print "Dictionaries loaded"
	
	n = len(dictNums)
	k = int(n/2)
	
	
	
	synsetObjects = [None] * n
	for key in dictNums:
		w = wn.synsets(dictNums[key])
		if len(w) > 0:
			synsetObjects[key] = (w[0])
			
	k = n/2
	simMatrix = lil_matrix((k,n))
	
	for i in xrange(0,k):
		simMatrix[i,i] = 3.6375861597263857 # maximum value of the lch_similarity()
	
	
	for i in xrange(0, k):
		rprint("\tProgress : %d%% (%d/%d)" % (100*i/n, i, n))
		if synsetObjects[i] is not None:
			w1 = synsetObjects[i]
			for j in xrange(0,n):
				if i != j and synsetObjects[j] is not None:
					w2 = synsetObjects[j]
					try:
						sim = w1.lch_similarity(w2)
						if sim > 1.5:
							simMatrix[i, j] = sim
					except:
						pass
				elif i == j:
					break
					
	save_sparse_lil("similarityMatrix1", simMatrix)
	save_sparse_lil_compressed("similarityMatrix1c", simMatrix)
	
	
	del simMatrix
	
	if n%2 == 1:
		l = k + 1
	else:
		l = k
	
	simMatrix = lil_matrix((l,n))
	
	for i in xrange(k,n):
		simMatrix[i-k,i] = 3.6375861597263857 # maximum value of the lch_similarity()
		
	for i in xrange(k,n):
		rprint("\tProgress : %d%% (%d/%d)" % (100*i/n, i, n))
		if synsetObjects[i] is not None:
			w1 = synsetObjects[i]
			for j in xrange(0,n):
				if i != j and synsetObjects[j] is not None:
					w2 = synsetObjects[j]
					try:
						sim = w1.lch_similarity(w2)
						if sim > 1.5:
							simMatrix[i-k, j] = sim
					except:
						pass
				elif i == j:
					break
					
	save_sparse_lil("similarityMatrix2", simMatrix)
	save_sparse_lil("similarityMatrix2c", simMatrix)
	
	print "Done	                                          "					
if __name__ == "__main__":
	main()