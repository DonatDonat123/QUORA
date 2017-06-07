from gensim.models import Word2Vec
import cPickle, gzip
import numpy as np
import pandas as pd

# Generates question embeddings from word embeddings by averaging them.

def w2v_features(model, row):
    q1_list = str(row['question1']).lower().split()
    q2_list = str(row['question2']).lower().split()
    return vectorize(model, q1_list), vectorize(model, q2_list)


def vectorize(model, words, num_features=300):
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    index2word_set = set(model.wv.index2word)
    
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])

    featureVec = np.divide(featureVec, nwords)
    return featureVec


def generate(model, df, filename):
    q = np.zeros((df.shape[0]*2,300), dtype="float32")
    i = 0
    for index, row in df.iterrows():
        if i % 10000 == 0:
            print i
        sol1, sol2 = w2v_features(model, row)
        q[i] = sol1
        q[i+df.shape[0]] = sol2
        i += 1

    np.savez_compressed(filename, q)


def main():
    model = Word2Vec.load("300features_40minwords_10context")
    print "Loaded model"
    
    df_train = pd.read_csv('data/train.csv')
    print "Loaded train set"

    df_test  = pd.read_csv('data/test.csv')
    print "Loaded test set"

    generate(model, df_train, 'embeddings_train.npz')
    print "Generated embeddings for train set"

    generate(model, df_test, 'embeddings_test.npz')
    print "Generated embeddings for test set"


if __name__ == "__main__":
    main()