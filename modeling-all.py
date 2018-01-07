import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from time import time

loc_train = 'input data.csv'
loc_current= 'current data.csv'
loc_train_output = 'cross validated predictions.csv'
loc_current_output = 'current predictions.csv'

df = pd.read_csv(loc_train, sep=',', encoding='ISO-8859-1')
current_df = pd.read_csv(loc_current, sep=',', encoding='ISO-8859-1')

#set aside unused columns when getting features
unused_cols = ['index', 'adjusted points', 'att bps', 'def bps', 'pas bps', 'cost', 'event id', 'mins', 'player id', 'player name', 'points', 'round', 'team id', 'transfers in', 'transfers out', 'seven plus', 'eight plus', 'nine plus', 'ten plus',  'adj points/transfers in', 'adj points/last 5', 'adj points/h adj last 3']
#comment out line below if you want to use binary opponent variables, for the upcoming season I will not because of major changes to team strengths
unused_cols.append(['opponent_' + str(i) for i in range(1, 23)])
features = [col for col in df.columns if col not in unused_cols]
target = 'adjusted points'
eight_classifier_target = 'eight plus'
ten_classifier_target = 'ten plus'
alternate_target = 'adj points/last 5'

#dimentionality reduction methods
dimensionality_reducers = [
('LocalPCA', PCA(n_components=2))]
#('LocaltSNE', TSNE(n_components=2))]

#TODO: Try LightGBM instead of Scikit's Gradient Boosting for improved speed
#lower level models
models = [
('GBM', GradientBoostingRegressor()),
('RF', RandomForestRegressor(n_estimators=15, min_samples_split=5)),
('Lasso', Lasso()),
('Ridge', Ridge()),
('RANSAC', RANSACRegressor()),
('TheilSen', TheilSenRegressor()),
('Huber', HuberRegressor()),
('KNN', KNeighborsRegressor(n_neighbors=10))]

#lower level classifiers for predicting 8+ points
eight_plus_classifiers = [
('NB eight plus', GaussianNB()),
('GBM eight plus', GradientBoostingClassifier()),
('RF eight plus', RandomForestClassifier(n_estimators=15, min_samples_split=5))]

ten_plus_classifiers = [
('NB ten plus', GaussianNB()),
('GBM ten plus', GradientBoostingClassifier()),
('RF ten plus', RandomForestClassifier(n_estimators=15, min_samples_split=5))]

#top level models
top_models = [
#('Top GBM', GradientBoostingRegressor()),
('Top Linear', LinearRegression()),
('Top Ridge', Ridge()),
('Top TheilSen', TheilSenRegressor()),
#('Top GBM 0.25', GradientBoostingRegressor(loss='quantile', alpha=0.25)),
#('Top GBM 0.75', GradientBoostingRegressor(loss='quantile', alpha=0.75)),
('Top GBM 0.30', GradientBoostingRegressor(loss='quantile', alpha=0.30))] #personally I find .30/.35 more useful as a tool fo measuring floor than .25

x = df[features]
current_x = current_df[features]
y = df[target]
y_clf_8 = df[eight_classifier_target]
y_clf_10 = df[ten_classifier_target]
y_alt = df[alternate_target]

#loop over dimentionality reduction methods
for name, reducer in dimensionality_reducers:
	t0 = time()
	combined_x = x.append(current_x) #can call fit_transform only once in tsne, so merge sets then divide
	transform = reducer.fit_transform(combined_x)
	split = df.shape[0]
	df[name + '1'] = transform[0:split,0]
	df[name + '2'] = transform[0:split,1]
	current_df[name + '1'] = transform[split:,0]
	current_df[name + '2'] = transform[split:,1]
	t1 = time()
	print('Dim. reduction %s: done in %f seconds' % (name, t1-t0))
	features.append(name + '1')
	features.append(name + '2')

#add the dimensionality reduced data to lower level input x
x = df[features]
current_x = current_df[features]

#loop over lower level models
for name, model in models:
	t0 = time()
	pred = cross_val_predict(model, x, y, cv=5)
	pred[(pred < 0) | (pred > 100)] = 0 #get rid of ridiculous values sometimes in linear regression
	df[name] = pred
	mae = mean_absolute_error(y, df[name])
	r2 = r2_score(y, df[name])
	model.fit(x, y)
	pred = model.predict(current_x)
	pred[(pred < 0) | (pred > 100)] = 0
	current_df[name] = pred
	t1 = time()
	print('Model %s: %f MAE, %f R^2 in %f seconds' % (name, mae, r2, t1-t0))
	features.append(name) #add model name to features used in top level models

#loop over lower level classifiers(for 8 plus)
for name, model in eight_plus_classifiers:
	t0 = time()
	df[name] = cross_val_predict(model, x, y_clf_8, cv=5, method='predict_proba')[:,0]
	auroc = roc_auc_score(y_clf_8, df[name])
	model.fit(x, y_clf_8)
	current_df[name] = model.predict_proba(current_x)[:,0]
	t1 = time()
	print('Model %s: %f AUROC in %f seconds' % (name, auroc, t1-t0))
	features.append(name) #add model name to features used in top level models

#loop over lower level classifiers(for 10 plus)
for name, model in ten_plus_classifiers:
	t0 = time()
	df[name] = cross_val_predict(model, x, y_clf_10, cv=5, method='predict_proba')[:,0]
	auroc = roc_auc_score(y_clf_10, df[name])
	model.fit(x, y_clf_10)
	current_df[name] = model.predict_proba(current_x)[:,0]
	t1 = time()
	print('Model %s: %f AUROC in %f seconds' % (name, auroc, t1-t0))
	features.append(name) #add model name to features used in top level models

#add the lower level model output reduced data to top level input x
x = df[features]
current_x = current_df[features]

#loop over top level models
for name, model in top_models:
	t0 = time()
	pred = cross_val_predict(model, x, y, cv=5)
	pred[(pred < 0) | (pred > 100)] = 0 #get rid of ridiculous values sometimes in linear regression
	df[name] = pred
	model.fit(x, y)
	pred = model.predict(current_x)
	pred[(pred < 0) | (pred > 100)] = 0
	current_df[name] = pred
	t1 = time()
	#don't evaluate model if it is for a prediction interval(Ex: Top GBM 0.25)
	if '.' not in name:
		mae = mean_absolute_error(y, df[name])
		r2 = r2_score(y, df[name])
		print('Model %s: %f MAE, %f R^2 in %f seconds' % (name, mae, r2, t1-t0))
	else:
		print('Model %s: in %f seconds' % (name, t1-t0))

	#used for testing the prediction of other variables
	#name = name + ' alt target'
	#model2 = clone(model)
	#t0 = time()
	#pred = cross_val_predict(model2, x, y_alt, cv=5)
	#pred[(pred < 0) | (pred > 100)] = 0 #get rid of ridiculous values sometimes in linear regression
	#df[name] = pred
	#mae = mean_absolute_error(y, df[name])
	#r2 = r2_score(y, df[name])
	#model2.fit(x, y)
	#pred = model2.predict(current_x)
	#pred[(pred < 0) | (pred > 100)] = 0
	#current_df[name] = pred
	#t1 = time()
	#print("Model %s: %f MAE, %f R^2 in %f seconds" % (name, mae, r2, t1-t0))

#write to csv
df.to_csv(loc_train_output, sep=',', index=False)
current_df.to_csv(loc_current_output, sep=',', index=False)