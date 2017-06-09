Paula's pipeline
====

1. embeddings.py: generates question embeddings from word embeddings
2. features.py: generates simple text features and question embedding features, stores everything to df_all.csv
3. xgbooster.py: trains linear model with boosting, parameters obtained with grid search, generates submission file

About Simple Neural Network + LSTM (Dennis)
=====

Execute first Helper Files + BOW at the Bottom
Then from Top to Bottom

I added an LSTM layer for each question before merging them, note the fast overfitting after the 2nd epoch already.
I had only 64 LSTM-units and a small train-set though, so maybe with more units and the whole train-set it will be better...
