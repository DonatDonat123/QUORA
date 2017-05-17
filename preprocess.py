import pandas as pd
import re
from sklearn.model_selection	import train_test_split
from common						import saveObj, rprint

def question_to_words(question , stops): # cleans a question
	words = re.sub("[^a-z]", " ", question.lower()).split();
	meaningful_words = [w for w in words if not w in stops]
	return( " ".join( meaningful_words ))

def clean_questions(questions, stops):
	cleaned = [];
	n, i = len(questions), 0
	for question in questions:
		rprint("\tProgress : %d%% (%d/%d)" % (100*i/n, i, n))
		cleaned.append(question_to_words(question, stops))
		i = i + 1
	return cleaned

def prep_train100(stops):
	print "Reading train data"
	train_raw = pd.read_csv("data/train.csv", encoding = "utf-8", na_filter = False)
	print "Cleaning first questions"
	train100_q1 = clean_questions(train_raw["question1"], stops)
	print "Cleaning second questions                           "
	train100_q2 = clean_questions(train_raw["question2"], stops)
	train100_du = train_raw["is_duplicate"]
	
	print "Saving 100%% train data                             "
	train100 = (train100_q1, train100_q2, train100_du);
	saveObj(train100, "preprocessed/train100.bin");
	return train100;
	
def prep_train_split(train100):
	print "Splitting train data" #one long command:
	train70_q1, train30_q1, \
	train70_q2, train30_q2, \
	train70_du, train30_du = train_test_split( \
	train100[0], train100[1], train100[2], train_size=0.7, random_state = 42);
	
	print "Saving 70%% train data"
	train70 = (train70_q1, train70_q2, train70_du)
	saveObj(train70, "preprocessed/train70.bin")
	print "Saving 30%% train data"
	train30 = (train30_q1, train30_q2, train30_du)
	saveObj(train30, "preprocessed/train30.bin")
	
def prep_train(stops):
	train100 = prep_train100(stops)
	prep_train_split(train100)
	
def prep_test(stops):
	print "Reading test data"
	test_raw = pd.read_csv("data/test.csv", encoding = "utf-8",  na_filter = False)
	print "Cleaning first questions"
	test_q1 = clean_questions(test_raw["question1"], stops)
	print "Cleaning second questions                           "
	test_q2 = clean_questions(test_raw["question2"], stops)
	test_id = test_raw["test_id"]
	
	print "Saving test data                                    "
	test = (test_q1, test_q2, test_id);
	saveObj(test, "preprocessed/test.bin");	

def prep_stops():
	print "Creating stopwords"
	from nltk.corpus				import stopwords
	orig_stops = set(stopwords.words("english"))
	not_stops = set(["what", "how", "who"]) # add more if you like
	stops = orig_stops - not_stops		# Set difference
	return stops

def main():
	stops = prep_stops()
	print stops
	prep_train(stops)
	#prep_test(stops)

if __name__ == "__main__":
	main()
