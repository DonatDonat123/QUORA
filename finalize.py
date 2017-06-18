from __future__ import division
import numpy as np
import pandas as pd
import xgboost as xgb
import datetime, time, operator
from common import rprint
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV

RS = 12357
#ROUNDS = 315*16 # 5040

#hyperparameters:
ROUNDS = 250
ETA = 0.05
p = 0.165
alpha = 0.25
beta = 1.5
mult = 0.8

np.random.seed(RS)

def cubicherm(x):
	a = alfa + beta - 2.0
	b = 3 - 2*alfa - beta
	return ((a*x + b)*x + alfa)*x*mult;

def main():
	rprint('Starting program')
	start  = time.time()
	params = {'objective':'binary:logistic', 'silent':1, 'seed':RS,
			  'eval_metric':'logloss', 'eta':ETA, 'max_depth':3,
			  'min_child_weight':3,'subsample':0.7}
	
	cv_params = {'learning_rate': [0.05, 0.1, 0.15, 0.2], 'subsample': [0.7, 0.8, 0.9],
				 'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}

	rprint('Reading Original Train data')
	df_train = pd.read_csv('data/train.csv')
	rprint('Reading Original Test data')
	df_test  = pd.read_csv('data/test.csv')
	rprint('Reading Simple Features')
	x = pd.read_csv('df_all.csv')
	
	rprint('Processing Simple Features')
	del x['w2vec_cos'], x['w2vec_min_dist'], x['w2vec_max_dist']
	x_train = x[:df_train.shape[0]]
	x_test  = x[df_train.shape[0]:]
	y_train = df_train['is_duplicate'].values
	test_id = df_test['test_id']
	del x, df_train, df_test
		
	rprint('Reading WordNet train Features')
	panni_train = pd.read_csv('panni_train.csv')
	rprint('Reading WordNet test Features')
	panni_test = pd.read_csv('panni_test.csv')
	rprint('Processing WordNet Features')
	# del panni_test['test_id']
	del panni_train['is_duplicate']
	x_train = x_train.join(panni_train);
	x_test = x_test.join(panni_test);
	del panni_test, panni_train
	
	rprint('Reading NN train Features')
	
	dennis_train = pd.read_csv('nn_features_training.csv')
	del dennis_train['id']
	print "train keys"
	print dennis_train.keys()
	x_train = x_train.join(dennis_train);
	del dennis_train
	
	rprint('Reading NN test Features')
	
	dennis_test = pd.read_csv('nn_features_test.csv')
	rprint('Processing NN test Features')
	del dennis_test['test_id']
	print "test keys"
	print dennis_test.keys()
	x_test = x_test.join(dennis_test);
	del dennis_test
	
	
	if 0: # Now we oversample the negative class - on your own risk of overfitting!
		pos_train = x_train[y_train == 1]
		neg_train = x_train[y_train == 0]

		print "Oversampling started for proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train)))
		#p = 0.165
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
	
	xtr_train, xtr_test, ytr_train, ytr_test = train_test_split(x_train, y_train, test_size=0.2, random_state=RS)
	del x_train, y_train
	g_train = xgb.DMatrix(xtr_train, label=ytr_train)
	del xtr_train, ytr_train
	g_test = xgb.DMatrix(xtr_test, label=ytr_test)
	del xtr_test, ytr_test
	watchlist  = [(g_train,'train'), (g_test,'eval')]
	
	rprint("Training XGBoost for {} rounds".format(ROUNDS))
	
	g_model = xgb.train(params, g_train, ROUNDS, watchlist)
	del g_train, g_test, watchlist
	
	rprint("Making predictions")
	tmp = xgb.DMatrix(x_test);
	del x_test
	preds = g_model.predict(tmp)

	#optimized_GBM = GridSearchCV(xgb.XGBClassifier(**params), cv_params, 
    #                        	 scoring='neg_log_loss', cv=5, n_jobs=-1, verbose=10)
	#optimized_GBM.fit(x_train, y_train)

	#print optimized_GBM.best_params_
	#preds = optimized_GBM.predict_proba(x_test)

	rprint("Writing output")
	sub = pd.DataFrame()
	sub['test_id'] = test_id
	sub['is_duplicate'] = preds #*.75 # BULL-HACK
	sub['is_duplicate'].apply(lambda x: cubicherm(x));
	sub.to_csv("xgb_seed{}_n{}_eta{}_gs_oversampled.csv".format(RS, ROUNDS, 0.05), index=False)
	rprint("Total time in seconds: {}".format(time.time() - start))


if __name__ == '__main__':
	main()