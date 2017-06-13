from common import *

dictWords = dict();
dictNums = dict();

def addfreqs(questions):
	i, n, wordnum = 0, len(questions), 0
	for question in questions:
		rprint("\tProgress : %d%% (%d/%d)" % (100*i/n, i, n))
		i = i + 1
		for word in question:
			if not dictWords.has_key(word):
				dictWords[word] = wordnum
				dictNums[wordnum] = word
				wordnum = wordnum + 1
	return wordnum

def main():
	wn = 0
	
	print "Loading preprocessed/train100.bin        "
	data = loadObj('preprocessed/train100.bin')
	print "  First questions                       "
	wn = wn + addfreqs(data[0])
	print "  Second questions                      "
	wn = wn + addfreqs(data[1])
	del data
	
	saveObj(dictWords, 'dictWords.bin');
	saveObj(dictNums, 'dictNums.bin');
	
	print "Done                                    "					
	
if __name__ == "__main__":
	main()