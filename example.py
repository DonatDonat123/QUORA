from common import loadObj, rprint
import numpy as np

def Question2Vec(word2vec, question):
	index2word_set = set(word2vec.index2word)
	avg, nwords = np.zeros(300), 0
	for word in question:
		if word in index2word_set:
			nwords = nwords + 1
			avg = np.add(avg, word2vec[word])
	avg = np.divide(avg,nwords+0.01)
	return avg
	
def sim_cos(a,b):
	from numpy import dot
	from numpy.linalg import norm
	return dot(a, b)/(norm(a)*norm(b)+0.000001)
	
def distanceOfQuestions(word2vec, question1, question2):
	q1vec = Question2Vec(word2vec, question1)
	q2vec = Question2Vec(word2vec, question2)
	return sim_cos(q1vec, q2vec);

def sortOnSim(word2vec, questions1, questions2):
	N = min(len(questions1), len(questions2));
	similarities = np.zeros(N);
	print "Calculating distances"
	for i in xrange(N):
		rprint("\tProgress : %d%% (%d/%d)" % (100*i/N, i, N))
		similarities[i] = distanceOfQuestions(word2vec, questions1[i], questions2[i])
	print "Sorting similarities"
	return sorted(range(N), key=lambda k: similarities[k])

def duplicateRatio():
	print "Loading train data"
	train = loadObj("preprocessed/train70.bin")
	N_train = len(train[0]);

	print "Calculating ratio of duplicates"
	num_of_duplicates = 0;
	for is_duplicate in train[2]:
		if is_duplicate:
			num_of_duplicates = num_of_duplicates + 1;
	ratio = num_of_duplicates / float(N_train)
	print "Ratio of duplicates = %f" % (ratio)
	return ratio
	
def activation_func(x, p): # parabolic interpolation
	a = 0.5 / (1 - p*p)
	return max(a * x*x + (1 - a) * x, 0) # nonnegative
	
def my_fit(sorted_inds, ratio):
	N = len(sorted_inds);
	result = np.zeros(N);
	for i in xrange(N):
		rprint("\tProgress : %d%% (%d/%d)" % (100*i/N, i, N))
		result[sorted_inds[i]] = activation_func(float(i/(N-1)), ratio)
	return result;
	
def calc_result(word2vec,data, ratio):
	sorted_inds = sortOnSim(word2vec, data[0], data[1]);
	print "Applying similarities"
	result = my_fit(sorted_inds, ratio);
	return result;
	
def main():
	print "Loading pretrained Word2Vec model"
	from gensim.models import Word2Vec
	word2vec = Word2Vec.load("300features_40minwords_10context")
	
	#ratio = duplicateRatio();
	ratio = 0.368639;
	
	print "Loading test data"
	test30 = loadObj("preprocessed/train30.bin")
	result = calc_result(word2vec,test30, ratio);
	
	print "Calculating log loss"
	from sklearn.metrics import log_loss
	error = log_loss(test30[2], result);
	print "Log loss = %f" % error
	
	test = loadObj("preprocessed/test.bin");
	result = calc_result(word2vec,test, ratio)
	
	from pandas import DataFrame
	output = pd.DataFrame( data={"test_id":test[2], "is_duplicate":result} )
	output.to_csv("submit.csv", index = False, quoting = 3)
	

if __name__ == "__main__":
	main()
