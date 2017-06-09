import numpy as np
import gzip, pickle as cPickle
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
# For Tensorflow & Keras
import tensorflow as tf
import keras.layers as lyr
from keras.models import Model
#For Padding
from keras.preprocessing.sequence import pad_sequences
#For XGBoost
import xgboost as xgb
#For Word2Vec
from gensim.models import Word2Vec
#from gensim.models import KeyedVectors

def main():
	# DEFINE GLOBAL VALUES
	STOPS = prep_stops()
	lstm_units = 256 
	num_voc = 5000
	EMBEDDING_DIM=300
	max_seq_length = 10
	#num_dense = 100
	rate_drop_lstm = 0.3
	#rate_drop_dense = 0.25


	# LOAD DATA
	X_train, X_traintest, Y_train, Y_traintest = splitData()
	X_test, test_idxs = testData()

	# 1st Stage, Question to Int Array
	# Build Vocabulary

	f1 = BoW()
	f1.fit(X_train.append(X_traintest), num_voc)
	X_all = X_train.append(X_traintest)
	mycv = f1.getVectorizer()
	#words to tokens
	X_test_q1 = create_padded_seqs(X_test.question1,mycv, max_seq_length)
	X_test_q2 = create_padded_seqs(X_test.question2, mycv, max_seq_length)
	X_train_q1 = create_padded_seqs(X_train.question1, mycv, max_seq_length)
	X_train_q2 = create_padded_seqs(X_train.question2, mycv, max_seq_length)
	X_traintest_q1 = create_padded_seqs(X_traintest.question1, mycv, max_seq_length)
	X_traintest_q2 = create_padded_seqs(X_traintest.question2, mycv, max_seq_length)

	# Create Embedding Matrix: Voc Entity --> Vector
	MODEL_FILE = './input/300features_40minwords_10context.bin'
	model = Word2Vec.load(MODEL_FILE)
	vocab = model.wv.vocab

	print "Loaded model"
	print('Preparing embedding matrix')

	nb_words = num_voc

	embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
	for word, i in mycv.vocabulary_.iteritems():
	    if word in vocab:
		embedding_matrix[i] = model[word]
	print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

	# Feed NN with Padded Sequences
	embedding_layer = lyr.Embedding(num_voc,
		EMBEDDING_DIM,
		weights=[embedding_matrix],
		input_length=max_seq_length,
		trainable=False)
	lstm_layer = lyr.LSTM(lstm_units, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

	input1_tensor = lyr.Input(X_train_q1.shape[1:])
	input2_tensor = lyr.Input(X_train_q2.shape[1:])
	e1 = embedding_layer(input1_tensor)
	e2 = embedding_layer(input2_tensor)
	lstm1 = lstm_layer(e1)
	lstm2 = lstm_layer(e2)
	merge_layer = lyr.concatenate([lstm1, lstm2])  ### MERGE LAYER
	#merge_layer = lyr.Dropout(rate_drop_dense)(merge_layer)
	#merge_layer = lyr.BatchNormalization()(merge_layer)
	#dense_layer = lyr.Dense(num_dense, activation='relu')(merge_layer) # DENSE LAYER
	#dense_layer = lyr.Dropout(rate_drop_dense)(dense_layer)
	#dense_layer = lyr.BatchNormalization()(dense_layer)
	output_layer = lyr.Dense(1, activation='sigmoid')(merge_layer)
	model = Model([input1_tensor, input2_tensor], output_layer)
	model.compile(loss='binary_crossentropy', optimizer='adam')
	model.summary()

	# FIT --> Train NN weights
	model.fit([X_train_q1, X_train_q2], Y_train, 
		  validation_data=([X_traintest_q1, X_traintest_q2], Y_traintest), 
		  batch_size=128, epochs=7, verbose=2)

	# (1) Take Features from Merge Layer and(2) feed to Classifier (XGBoost)
	#(1)
	mergeNN = Model([input1_tensor, input2_tensor], merge_layer)
	mergeNN.compile(loss='mse', optimizer='adam')

	F_train = mergeNN.predict([X_train_q1, X_train_q1], batch_size=128)
	F_traintest = mergeNN.predict([X_traintest_q1, X_traintest_q2], batch_size=128)
	F_test = mergeNN.predict([X_test_q1, X_test_q2], batch_size=128)

	#(2)
	dTrain = xgb.DMatrix(F_train, label=Y_train)
	dTraintest = xgb.DMatrix(F_traintest, label=Y_traintest)
	dTest = xgb.DMatrix(F_test)

	xgb_params = {
	    'objective': 'binary:logistic',
	    'booster': 'gbtree',
	    'eval_metric': 'logloss',
	    'eta': 0.1, 
	    'max_depth': 9,
	    'subsample': 0.9,
	    'colsample_bytree': 1 / F_train.shape[1]**0.5,
	    'min_child_weight': 5,
	    'silent': 1
	}
	bst = xgb.train(xgb_params, dTrain, 1000,  [(dTrain,'train'), (dTraintest,'val')], 
		        verbose_eval=10, early_stopping_rounds=10)


	# PREDICT Submission File
	dTest = xgb.DMatrix(F_test)
	df_sub = pd.DataFrame({
		'test_id': test_idxs,
		'is_duplicate': bst.predict(dTest, ntree_limit=bst.best_ntree_limit)
	    }).set_index('test_id')


	print("Create Submission File")
	df_sub.to_csv('newsub.csv')


##########HELPER FUNCTIONS#############
def prep_stops():
	print "Creating stopwords"
	from nltk.corpus import stopwords
	orig_stops = set(stopwords.words("english"))
	not_stops = set(["what", "how", "who"]) # add more if you like
	stops = orig_stops - not_stops		# Set difference
	return stops

def create_padded_seqs(texts, mycv,max_seq_length):
    seqs = texts.apply(text2ints, cv=mycv)
    return pad_sequences(seqs, maxlen=max_seq_length)

def text2ints(text, cv):
    other_index = len(cv.vocabulary_)
    intseq = []
    mysplit = text.split(" ")
    filter(lambda x: x!="", mysplit)
    for word in mysplit:
        intseq.append(cv.vocabulary_[word]) if word in cv.vocabulary_ else other_index
    return intseq

def load_train():
    #Train_Path = './input/df10.csv' # JUST TO TEST FASTER , normally train.csv
    Train_Path = './input/train.csv'
    print("LOAD TRAIN DATA")
    train = pd.read_csv(Train_Path)
    train['question1'].fillna('', inplace=True)
    train['question2'].fillna('', inplace=True)
    train = cleanquestions(train)
    return train
def load_test():
    #Test_Path = './input/test1000.csv' # JUST TO TEST FASTER
    Test_Path = './input/test.csv'
    print("LOAD TEST DATA")
    test = pd.read_csv(Test_Path)
    test['question1'].fillna('', inplace=True)
    test['question2'].fillna('', inplace=True)
    test = cleanquestions(test)
    return test

def cleanreview(review):
    STOPS = [""]
    review = re.sub("[^a-z]", " ", review.lower()) # letters only
    review = review.split(" ")
    meaningful_words = [w for w in review if not w in STOPS]
    sentence = " ".join(meaningful_words)
    return sentence
    
def cleanquestions(df):

    question1 = df.question1.apply(lambda x: cleanreview(x))
    question2 = df.question2.apply(lambda x: cleanreview(x))
    df.loc[:, 'question1'] = pd.Series(question1, index=df.index)
    df.loc[:, 'question2'] = pd.Series(question1, index=df.index)
    return df

def splitData(ratio = 0.7): # ratio = train_set, between 0-1.0, default 0.7
    print("SPLIT DATA")
    #time.sleep(1)
    train = load_train()
    X_train, X_test, Y_train, Y_test = train_test_split(train.loc[:,["question1", "question2"]],\
                                                        train["is_duplicate"],\
                                                        train_size=0.7, random_state=42)
    return X_train, X_test, Y_train, Y_test
def trainData():
    print("LOAD TRAIN DATA FOR PREDICT AND SUBMIT")
    train = load_train()
    X = train.loc[:, ["question1", "question2"]]
    Y = train.loc[:]["is_duplicate"]
    return X,Y
def testData():
    test = load_test()
    X_test = test.loc[:,["question1", "question2"]]
    X_ids = test.loc[:]["test_id"]
    return X_test, X_ids

def rprint(str): # Next print overwrites this, i.e use for indicate progress
	sys.stdout.write("PROCESSING: " + str + "            \r")
	sys.stdout.flush()


# serializes data and saves in a (somewhat) compressed format
def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        cPickle.dump(obj, f, protocol)
        f.close()

# loads compressed data and restores original state
def load_zipped_pickle(filename):	# loads and unpacks
    with gzip.open(filename, 'rb') as f:
        loaded_object = cPickle.load(f)
        f.close()
        return loaded_object   
    
    
class BoW:
    name = "BoW"
    cv = 0

    def fit(self, X, num_vocab):
        corpus = pd.concat([X.question1, X.question2])
        self.cv = CountVectorizer(analyzer='word',min_df = 0,
        max_features=num_vocab, ngram_range=(1,1), preprocessor=None, stop_words=None,
        tokenizer=None)
        print("FIT CORPUS ON COUNT VECTORIZER")
        self.cv.fit(corpus)

    def transform(self, X):
        return self.cv.transform(X)
    def getVectorizer(self):
        return self.cv

if __name__ == "__main__":
    main()
