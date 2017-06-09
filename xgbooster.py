from __future__ import division
import numpy as np
import pandas as pd
import xgboost as xgb
import datetime
import operator
from sklearn.cross_validation import train_test_split
import time
from sklearn.model_selection import GridSearchCV

RS = 12357
ROUNDS = 315*16 # 5040
ETA = 0.11

np.random.seed(RS)


def train_xgb(X, y, params):
	print("Will train XGB for {} rounds, RandomSeed: {}".format(ROUNDS, RS))
	x, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RS)

	xg_train = xgb.DMatrix(x, label=y_train)
	xg_val = xgb.DMatrix(X_val, label=y_val)

	watchlist  = [(xg_train,'train'), (xg_val,'eval')]
	return xgb.train(params, xg_train, ROUNDS, watchlist)


def predict_xgb(clr, X_test):
	return clr.predict(xgb.DMatrix(X_test))


def main():
	start  = time.time()
	params = {}
	params['objective'] = 'binary:logistic'
	params['silent'] = 1
	params['seed'] = RS

	params['eval_metric'] = 'logloss'
	params['eta'] = 0.05
	params['max_depth'] = 3
	params['min_child_weight'] = 3
	params['subsample'] = 0.7
	
	cv_params = {'learning_rate': [0.05, 0.1, 0.15, 0.2], 'subsample': [0.7, 0.8, 0.9],
				 'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}

	df_train = pd.read_csv('train.csv')
	df_test  = pd.read_csv('test.csv')
	x = pd.read_csv('df_all.csv')
	del x['w2vec_cos']
	del x['w2vec_min_dist']
	del x['w2vec_max_dist']

	x_train = x[:df_train.shape[0]]
	x_test  = x[df_train.shape[0]:]
	y_train = df_train['is_duplicate'].values
	del x, df_train

	if 1: # Now we oversample the negative class - on your own risk of overfitting!
		pos_train = x_train[y_train == 1]
		neg_train = x_train[y_train == 0]

		print "Oversampling started for proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train)))
		p = 0.165
		scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
		while scale > 1:
			neg_train = pd.concat([neg_train, neg_train])
			scale -=1
		neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
		print "Oversampling done, new proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train)))

		x_train = pd.concat([pos_train, neg_train])
		y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
		del pos_train, neg_train

	print "Training data: X_train: {}, Y_train: {}, X_test: {}".format(x_train.shape, len(y_train), x_test.shape)
	clr = train_xgb(x_train, y_train, params)
	preds = predict_xgb(clr, x_test)

	#optimized_GBM = GridSearchCV(xgb.XGBClassifier(**params), cv_params, 
    #                        	 scoring='neg_log_loss', cv=5, n_jobs=-1, verbose=10)
	#optimized_GBM.fit(x_train, y_train)

	#print optimized_GBM.best_params_
	#preds = optimized_GBM.predict_proba(x_test)

	print "Writing output..."
	sub = pd.DataFrame()
	sub['test_id'] = df_test['test_id']
	sub['is_duplicate'] = preds *.75
	sub.to_csv("xgb_seed{}_n{}_eta{}_gs_oversampled.csv".format(RS, ROUNDS, 0.05), index=False)

	print "Total time in seconds: {}".format(time.time() - start)


if __name__ == '__main__':
	main()