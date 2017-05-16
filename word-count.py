import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from nltk import word_tokenize
from collections import Counter
import pickle


# Simple word-counting classification. Steps included:
# 1) Load the training data, clean it and tokenize.
# 2) Count how many words in common two questions have.
# 3) Optimize a threshold on the training set.
#    If the no. of words in common is less than the threshold,
#    the questions are not considered duplicates, and vice-versa.
# 4) Use the best-scoring threshold on the training set on the
#    test set and produce a submission file.


def filter_words(text):
    punctuation = [',','.',':',';','!','?','-', '...']
    stopwords = ['do', 'the', 'a', 'an']
    try:
        tokens = word_tokenize(text.lower())
    except:
        print text
        tokens = []
    return Counter(filter(lambda x: not x in punctuation and not x in stopwords, tokens))


def common_words(text1, text2):
    f1, f2 = filter_words(text1), filter_words(text2)
    common = f1 & f2
    return sum(common.values())


def classify(threshold, train):
    predicted = []
    i = 0
    for index, row in train.iterrows():
        if i % 100000 == 0 and i > 0:
            print i
        intersect = common_words(row["question1"], row["question2"])
        predicted.append(intersect)
        i += 1
        # print intersect, test.loc[index]
    
    pred = [0 if p < threshold else 1 for p in predicted]
    return pred


def get_best_threshold(thresholds, X_train, Y_train):    
    best_score = 100000.
    best_t = 0
    for t in thresholds:
        pred = classify(t, X_train)
        curr_score = log_loss(Y_train, pred)
        if curr_score < best_score:
            best_t = t
            best_score = curr_score
        
    return best_t, best_score


def make_submission(final_preds, ids):
    sub = pd.DataFrame(data={"test_id":ids,"is_duplicate":final_preds})
    sub.to_csv('sub.csv', columns=['test_id', 'is_duplicate'], index=False)


def main():
    data = pd.read_csv('train.csv', encoding="utf-8")

    X_train, X_test, Y_train, Y_test = train_test_split(data.loc[:,["question1", "question2"]], data["is_duplicate"], train_size=0.7, random_state=42)

    thresholds = range(2, 100)
    t, score = get_best_threshold(thresholds, X_train, Y_train)
    print "Best threshold: {0}".format(t)

    test_data = pd.read_csv('test.csv', encoding="utf-8")
    final_preds = classify(t, test_data)

    with open("final_preds.pickle",'wb') as f:
        pickle.dump(final_preds, f)

    make_submission(final_preds, test_data["test_id"])


if __name__ == "__main__":
    main()
