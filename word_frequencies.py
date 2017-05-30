from common import *

dictionary = dict();

def addfreqs(questions):
	i, n, wordnum = 0, len(questions), 0
	for question in questions:
		rprint("\tProgress : %d%% (%d/%d)" % (100*i/n, i, n))
		i = i + 1
		for word in question:
			if dictionary.has_key(word):
				dictionary[word] = dictionary[word] + 1
				wordnum = wordnum + 1
	return wordnum

def main():
	print "Loading set of words used by Word2Vec"
	
	from gensim.models import Word2Vec
	word2vec = Word2Vec.load("300features_40minwords_10context")
	for word in word2vec.index2word:
		dictionary[word] = 0
	
	wn = 0
	
	print "Loading preprocessed/train30.bin        "
	data = loadObj('preprocessed/train30.bin')
	print "  First questions                       "
	wn = wn + addfreqs(data[0])
	print "  Second questions                      "
	wn = wn + addfreqs(data[1])
	del data
	
	print "Loading preprocessed/train70.bin        "
	data = loadObj('preprocessed/train70.bin')
	print "  First questions                       "
	wn = wn + addfreqs(data[0])
	print "  Second questions                      "
	wn = wn + addfreqs(data[1])
	del data
	
	print "Loading preprocessed/test.bin           "
	data = loadObj('preprocessed/test.bin')
	print "  First questions                       "
	wn = wn + addfreqs(data[0])
	print "  Second questions                      "
	wn = wn + addfreqs(data[1])
	del data
	
	for key in dictionary:
		print key, dictionary[key]
	
	saveObj(dictionary, 'frequencies_dictionary.bin');
	
	print "Done                                    "					
	
if __name__ == "__main__":
	main()