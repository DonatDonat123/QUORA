import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from nltk import word_tokenize
from collections import Counter
import pickle
from sklearn.linear_model import LogisticRegression


# Simple word-counting classification. Steps included:
# 1) Load the training data, clean it and tokenize.
# 2) Count how many words in common two questions have,
#    averaged by question length.
# 3) Train a logistic regression model for classification.


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
    return sum(common.values()), sum(f1.values()), sum(f2.values())


def featurize(X):
    features = []
    i = 0
    for index, row in X.iterrows():
        if i % 100000 == 0 and i > 0:
            print i
        i += 1
        intersect, l_q1, l_q2 = common_words(row["question1"], row["question2"])
        q_length = min(l_q1, l_q2)
        sol = intersect / float(q_length) if q_length > 0 else 0
        features.append([sol])
    
    return features


def make_submission(final_preds, ids):
    sub = pd.DataFrame(data={"test_id":ids,"is_duplicate":final_preds})
    sub.to_csv('sub.csv', columns=['test_id', 'is_duplicate'], index=False)


def main():
    data = pd.read_csv('train.csv', encoding="utf-8")

    X_train, X_test, Y_train, Y_test = train_test_split(data.loc[:,["question1", "question2"]], data["is_duplicate"], train_size=0.7, random_state=42)

    test_data = pd.read_csv('test.csv', encoding="utf-8")
    print "Done loading data!"
    
    f_train = featurize(X_train)
    print "Done featurizing train!"
    f_test = featurize(test_data)
    print "Done featurizing test!"

    logistic = LogisticRegression()
    logistic.fit_transform(f_train, Y_train)

    print "Done fitting model!"

    final_preds = map(lambda x: x[1], logistic.predict_proba(f_test))

    with open("final_preds.pickle",'wb') as f:
        pickle.dump(final_preds, f)

    make_submission(final_preds, test_data["test_id"])


if __name__ == "__main__":
    main()
