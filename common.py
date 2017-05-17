import cPickle, gzip, sys

def saveObj(obj, filename):
	with gzip.open(filename, 'wb') as f:
		cPickle.dump(obj, f, -1)

def loadObj(filename):
	with gzip.open(filename, 'rb') as f:
		loaded_object = cPickle.load(f)
		return loaded_object

def rprint(str): # Next print overwrites this, eg. use to indicate progress
	sys.stdout.write(str + "               \r")
	sys.stdout.flush()