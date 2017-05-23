from common import loadObj, saveObj, rprint
import numpy as np

wordvector_dimensionality = 300

#avg is modified but not reassigned
def question2Vec(question, avg, word2vec, index2word_set):
	nwords = 0
	for word in question:
		if word in index2word_set:
			nwords = nwords + 1
			avg += word2vec[word]	# modifies array in-place
	avg /= nwords+0.01				# this too
	
def convertQuestions2Mat(questions, word2vec, index2word_set):
	N, i = len(questions), 0
	vectors = np.zeros((N, wordvector_dimensionality), dtype = np.float32)
	for question in questions:
		rprint("\tProgress : %d%% (%d/%d)" % (100*i/N, i, N))
		question2Vec(question,vectors[i], word2vec, index2word_set);
		i = i + 1
	return vectors
	
def convertFile2VecFile(inputpath, outputpath, word2vec, index2word_set):
	print "Loading " + inputpath
	data = loadObj(inputpath);
	print "  Processing first questions"
	Q1s = convertQuestions2Mat(data[0], word2vec, index2word_set)
	print "  Processing second questions              "
	Q2s = convertQuestions2Mat(data[1], word2vec, index2word_set)
	print "Saving to " + outputpath + "                "
	saveObj((Q1s,Q2s,data[2]), outputpath)

# a memory optimized version of the previous function:
def convertBigFile2VecBigFiles(inputpath, outputpath, word2vec, index2word_set):
	print "Loading " + inputpath
	data = loadObj(inputpath);
	inQ1s, inQ2s, t3rd = data[0], data[1], data[2]
	del data
	
	print "  Processing first questions"
	Q1s = convertQuestions2Mat(inQ1s, word2vec, index2word_set)
	del inQ1s
	print "  Saving first questions to " + outputpath + '0.bin'
	np.savez_compressed(outputpath + '0.npz', Q1s);
	#saveObj(Q1s, outputpath + '0.bin')
	del Q1s
	
	print "  Processing second questions              "
	Q2s = convertQuestions2Mat(inQ2s, word2vec, index2word_set)
	del inQ2s
	print "  Saving second questions to " + outputpath + '1.bin'
	np.savez_compressed(outputpath + '1.npz', Q2s);
	#saveObj(Q2s, outputpath + '1.bin')
	del Q2s
	
	print "  Saving third data values to " + outputpath + '2.bin'
	saveObj(t3rd, outputpath + '2.bin')
	
def main():
	print "Loading pretrained Word2Vec model"
	from gensim.models import Word2Vec
	word2vec = Word2Vec.load("300features_40minwords_10context")
	index2word_set = set(word2vec.index2word)
	
	convertFile2VecFile("preprocessed/train30.bin", "word2vec/train30.bin", \
						word2vec, index2word_set)
	
	convertFile2VecFile("preprocessed/train70.bin", "word2vec/train70.bin", \
						word2vec, index2word_set)
	convertFile2VecFile("preprocessed/train100.bin", "word2vec/train100.bin", \
						word2vec, index2word_set)
	# This results in a MemoryError!
	#convertFile2VecFile("preprocessed/test.bin", "word2vec/test.bin", \
	#					word2vec, index2word_set)
	convertBigFile2VecBigFiles("preprocessed/test.bin", "word2vec/test", \
								word2vec, index2word_set)
	
	print "Done"					
	
if __name__ == "__main__":
	main()