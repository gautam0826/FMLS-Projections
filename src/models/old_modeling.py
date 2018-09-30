import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import inspect
import warnings
import mord
#from lasagne.updates import nesterov_momentum
#from lasagne.nonlinearities import softmax
#from lasagne.layers import DenseLayer
#from lasagne.layers import InputLayer
#from nolearn.lasagne import NeuralNet
from gplearn.genetic import SymbolicRegressor
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.exceptions import ConvergenceWarning
from time import time
pd.options.mode.chained_assignment = None  #gets rid of SettingWithCopyWarnings
warnings.filterwarnings("ignore", category=ConvergenceWarning) #get rid of Convergence Warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) #get rid of Deprecation Warnings

loc_train = '2018 input data.csv'
loc_current= '2018 current data.csv'
loc_train_output = '2018 cross validated predictions.csv'
loc_current_output = '2018 current predictions.csv'

df = pd.read_csv(loc_train, sep=',', encoding='ISO-8859-1')
current_df = pd.read_csv(loc_current, sep=',', encoding='ISO-8859-1')

#set aside unused columns when getting features
unused_cols = ['index', 'adjusted points', 'att bps', 'def bps', 'pas bps', 'cost', 'event id', 'mins', 'player id', 'player name', 'points', 'round', 'team id', 'transfers in', 'transfers out', 'seven plus', 'eight plus', 'nine plus', 'ten plus',  'adj points/transfers in', 'adj points/last 5', 'adj points/h adj last 3', 'position id_1', 'position id_2', 'position id_3', 'position id_4', 'season']
#comment out line below if you want to use binary opponent variables, for the upcoming season I will not because of major changes to team strengths
unused_cols.extend([col for col in df.columns if 'opponent' in col or 'position' in col])
features = [col for col in df.columns if col not in unused_cols]
target = 'adjusted points'
transformed_target = 'transformed adj points'
eight_classifier_target = 'eight plus'
ten_classifier_target = 'ten plus'

#dimentionality reduction methods
dimensionality_reducers = [
('LocalPCA', PCA(n_components=2))]
#('LocaltSNE', TSNE(n_components=2))]

#lower level models
gk_models = [
('Ordinal Ridge', mord.OrdinalRidge()),
('LAD', mord.LAD()),
#('NNPipe', Pipeline(steps=[('Scaler', StandardScaler()), ('Neural Network', MLPRegressor(max_iter = 250, hidden_layer_sizes = (50)))])),
#('SVMPipe', Pipeline(steps=[('SVD', TruncatedSVD()), ('SVM', SVR(kernel='poly', C=1))])),
('SVM', LinearSVR(C=0.01, dual=True, epsilon=1.0, loss='epsilon_insensitive', tol=1e-05)),
('ET', ExtraTreesRegressor(bootstrap=False, min_samples_leaf=10, min_samples_split=20, n_estimators=100)),
('ET2', ExtraTreesRegressor(bootstrap=False, min_samples_leaf=15, min_samples_split=15, n_estimators=100)),
('Ridge', Ridge(alpha=1)),
('RANSAC', RANSACRegressor()),
#('TheilSen', TheilSenRegressor()),
('ElasticNet', ElasticNet(l1_ratio=.9, alpha=1)),
('Huber', HuberRegressor(epsilon=1.5)),
('Genetic', SymbolicRegressor(population_size=200)),
#('Nnet', NeuralNet(layers=[('input', layers.InputLayer), ('hidden1', layers.DenseLayer), ('hidden2', layers.DenseLayer), ('output', layers.DenseLayer)],
#	input_shape=(None, len(features)), hidden1_num_units=50, hidden2_num_units=50, output_num_units=1, update=nesterov_momentum, update_learning_rate = 0.01)),
('KNN', KNeighborsRegressor(n_neighbors=10))]

df_models = [
('Ordinal Ridge', mord.OrdinalRidge()),
('LAD', mord.LAD()),
('GBM', GradientBoostingRegressor()),
('SVM', LinearSVR(C=0.01, dual=True, epsilon=1.0, loss='epsilon_insensitive', tol=1e-05)),
#('RF', RandomForestRegressor(bootstrap=False, min_samples_leaf=10, min_samples_split=15, n_estimators=100)),
('ET', ExtraTreesRegressor(bootstrap=False, min_samples_leaf=10, min_samples_split=20, n_estimators=100)),
#('ET2', ExtraTreesRegressor(bootstrap=False, min_samples_leaf=15, min_samples_split=15, n_estimators=100)),
('Ridge', Ridge(alpha=1)),
('RANSAC', RANSACRegressor()),
#('ElasticNet', ElasticNet(l1_ratio=.9, alpha=1)),
('Genetic', SymbolicRegressor(population_size=200)),
('Huber', HuberRegressor(epsilon=1.5))]

mf_models = [
('Ordinal Ridge', mord.OrdinalRidge()),
('LAD', mord.LAD()),
('SVM', LinearSVR(C=0.01, dual=True, epsilon=1.0, loss='epsilon_insensitive', tol=1e-05)),
('ET', ExtraTreesRegressor(bootstrap=False, min_samples_leaf=10, min_samples_split=20, n_estimators=100)),
('ET2', ExtraTreesRegressor(bootstrap=False, min_samples_leaf=15, min_samples_split=15, n_estimators=100)),
('NN', MLPRegressor(max_iter = 250, hidden_layer_sizes = (50, 50))),
('GBM', GradientBoostingRegressor()),
('Ridge', Ridge(alpha=1)),
#('ARD', ARDRegression()),
('RANSAC', RANSACRegressor()),
('ElasticNet', ElasticNet(l1_ratio=.9, alpha=1)),
('Genetic', SymbolicRegressor(population_size=200)),
('Huber', HuberRegressor(epsilon=1.5))]

fw_models = [
#('MordAT', mord.LogisticAT()),
#('MordIT', mord.LogisticIT()),
#('MordSE', mord.LogisticSE()),
('Ordinal Ridge', mord.OrdinalRidge()),
('LAD', mord.LAD()),
('SVM', LinearSVR(C=0.01, dual=True, epsilon=1.0, loss='epsilon_insensitive', tol=1e-05)),
('ET', ExtraTreesRegressor(bootstrap=False, min_samples_leaf=10, min_samples_split=20, n_estimators=100)),
('ET2', ExtraTreesRegressor(bootstrap=False, min_samples_leaf=15, min_samples_split=15, n_estimators=100)),
('GBM', GradientBoostingRegressor()),
('Ridge', Ridge(alpha=1)),
('RANSAC', RANSACRegressor()),
#('ARD', ARDRegression()),
#('TheilSen', TheilSenRegressor()),
('ElasticNet', ElasticNet(l1_ratio=.3, alpha=1)),
('Genetic', SymbolicRegressor(population_size=200)),
('Huber', HuberRegressor(epsilon=1.5))]

#lower level classifiers for predicting 8+ and 10+ points
classifiers = [
('NB', GaussianNB()),
('GBM', GradientBoostingClassifier()),
('Logistic1', LogisticRegression(C=1)),
('Logistic2', LogisticRegression(C=100)),
('RF', RandomForestClassifier(n_estimators=15, min_samples_split=5))]

#top level models
top_gk_models = [
('Top GBM', GradientBoostingRegressor()),
('Top SVM', LinearSVR(C=0.01, dual=True, epsilon=1.0, loss='epsilon_insensitive', tol=1e-05)),
('Top Ridge', Ridge(alpha=1)),
('Top Huber', HuberRegressor(epsilon=1.5)),
('Top Genetic', SymbolicRegressor(population_size=100)),
('Top Ordinal Ridge', mord.OrdinalRidge()),
#('Top TheilSen', TheilSenRegressor()),
#('Top GBM 0.75', GradientBoostingRegressor(loss='quantile', alpha=0.75)), ceiling value not necessary
('Top GBM 0.25', GradientBoostingRegressor(loss='quantile', alpha=0.25))]

top_df_models = [
('Top GBM', GradientBoostingRegressor()),
('Top Ridge', Ridge(alpha=1)),
('Top Huber', HuberRegressor(epsilon=1.5)),
('Top Genetic', SymbolicRegressor(population_size=100)),
('Top Ordinal Ridge', mord.OrdinalRidge()),
('Top GBM 0.25', GradientBoostingRegressor(loss='quantile', alpha=0.25))]

top_mf_models = [
('Top GBM', GradientBoostingRegressor()),
('Top SVM', LinearSVR(C=0.01, dual=True, epsilon=1.0, loss='epsilon_insensitive', tol=1e-05)),
('Top Ridge', Ridge(alpha=1)),
('Top Huber', HuberRegressor(epsilon=1.5)),
('Top Genetic', SymbolicRegressor(population_size=100)),
('Top Ordinal Ridge', mord.OrdinalRidge()),
('Top GBM 0.25', GradientBoostingRegressor(loss='quantile', alpha=0.25))]

top_fw_models = [
('Top GBM', GradientBoostingRegressor()),
('Top SVM', LinearSVR(C=0.01, dual=True, epsilon=1.0, loss='epsilon_insensitive', tol=1e-05)),
('Top Ridge', Ridge(alpha=1)),
('Top Huber', HuberRegressor(epsilon=1.5)),
('Top Genetic', SymbolicRegressor(population_size=100)),
('Top Ordinal Ridge', mord.OrdinalRidge()),
('Top GBM 0.25', GradientBoostingRegressor(loss='quantile', alpha=0.25))]

position_specific_models = [
('position id_1.0', gk_models, top_gk_models),
('position id_2.0', df_models, top_df_models),
('position id_3.0', mf_models, top_mf_models),
('position id_4.0', fw_models, top_fw_models)]

final_df = pd.DataFrame()
final_current_df = pd.DataFrame()

for position_name, models, top_models in position_specific_models:
	print(position_name)

	#select only the position being modeled
	position_df = df[df[position_name] == 1]
	position_current_df = current_df[current_df[position_name] == 1]

	#transform adjusted points
	transformer = QuantileTransformer(output_distribution='normal')
	#transformer = RobustScaler(quantile_range=(20, 80))
	position_df[transformed_target] = transformer.fit_transform(position_df['adjusted points'].values.reshape(-1, 1))

	#reset features variable each time so the previous position's additional feature columns don't get mixed in
	features = [col for col in df.columns if col not in unused_cols]
	x = position_df[features]
	current_x = position_current_df[features]
	y = position_df[target]
	y_clf_8 = position_df[eight_classifier_target]
	y_clf_10 = position_df[ten_classifier_target]
	y_trf = position_df[transformed_target]

	#set up column for final model with best mae and column with average of top level models weighted by mae
	best_mae_col_name = 'player id' #initialize with something with no correlation
	best_mae = 100 #initialize with really high value
	total_inv_mae = 0
	position_df['Top Models Weighted Avg'] = 0
	position_current_df['Top Models Weighted Avg'] = 0

	#loop over dimentionality reduction methods
	for name, reducer in dimensionality_reducers:
		t0 = time()
		combined_x = x.append(current_x) #can call fit_transform only once in tsne, so merge sets then divide
		transform = reducer.fit_transform(combined_x)
		split = position_df.shape[0]
		position_df[name + '1'] = transform[0:split,0]
		position_df[name + '2'] = transform[0:split,1]
		position_current_df[name + '1'] = transform[split:,0]
		position_current_df[name + '2'] = transform[split:,1]
		t1 = time()
		print('Dim. reduction %s: done in %f seconds' % (name, t1-t0))
		features.append(name + '1')
		features.append(name + '2')

	#add the dimensionality reduced data to lower level input x
	x = position_df[features]
	current_x = position_current_df[features]

	#loop over lower level models
	for name, model in models:
		model = clone(model) #don't want affect model for other positions
		t0 = time()
		pred = cross_val_predict(model, x, y, cv=5)
		pred[(pred < 0) | (pred > 100)] = 0 #get rid of ridiculous values sometimes in linear regression
		position_df[name] = pred
		mae = mean_absolute_error(y, position_df[name])
		r2 = r2_score(y, position_df[name])
		model.fit(x, y)
		pred = model.predict(current_x)
		pred[(pred < 0) | (pred > 100)] = 0
		position_current_df[name] = pred
		t1 = time()
		print('Model %s: %f MAE, %f R^2 in %f seconds' % (name, mae, r2, t1-t0))
		features.append(name) #add model name to features used in top level models

	#loop over lower level classifiers(for 8 plus)
	for name, model in classifiers:
		model = clone(model) #don't want affect model for other positions
		name += ' eight plus'
		t0 = time()
		position_df[name] = cross_val_predict(model, x, y_clf_8, cv=5, method='predict_proba')[:,1]
		auroc = roc_auc_score(y_clf_8, position_df[name])
		model.fit(x, y_clf_8)
		position_current_df[name] = model.predict_proba(current_x)[:,1]
		t1 = time()
		print('Model %s: %f AUROC in %f seconds' % (name, auroc, t1-t0))
		features.append(name) #add model name to features used in top level models

	#loop over lower level classifiers(for 10 plus)
	for name, model in classifiers:
		model = clone(model) #don't want affect model for other positions
		name += ' ten plus'
		t0 = time()
		position_df[name] = cross_val_predict(model, x, y_clf_10, cv=5, method='predict_proba')[:,1]
		auroc = roc_auc_score(y_clf_10, position_df[name])
		model.fit(x, y_clf_10)
		position_current_df[name] = model.predict_proba(current_x)[:,1]
		t1 = time()
		print('Model %s: %f AUROC in %f seconds' % (name, auroc, t1-t0))
		features.append(name) #add model name to features used in top level models

	#remove some lower-level features
	removed_cols = ['DGW', 'Genetic', 'last 3 adjusted points avg', 'adjusted points weighted avg', 'season adjusted points avg', 'home']
	removed_cols.extend([feature for feature in features if 'cluster' in feature or 'h adj' in feature or 'PCA' in feature])
	features = [feature for feature in features if feature not in removed_cols]

	#add the lower level model output reduced data to top level input x
	x = position_df[features]
	current_x = position_current_df[features]

	#loop over top level models
	for name, model in top_models:
		t0 = time()
		pred = cross_val_predict(model, x, y, cv=5)
		pred[(pred < 0) | (pred > 100)] = 0 #get rid of ridiculous values sometimes in linear regression
		position_df[name] = pred
		model.fit(x, y)
		pred = model.predict(current_x)
		pred[(pred < 0) | (pred > 100)] = 0
		position_current_df[name] = pred
		t1 = time()
		#don't evaluate model if it is for a prediction interval(Ex: Top GBM 0.25)
		if '.' not in name and 'Ordinal Ridge' not in name:
			#calculate metrics
			mae = mean_absolute_error(y, position_df[name])
			r2 = r2_score(y, position_df[name])
			best_mae = mean_absolute_error(y, position_df[best_mae_col_name])

			#add up inverse maes (instead of mae since we want to give higher maes a higher weight)
			inv_mae = 1 / mae
			position_df['Top Models Weighted Avg'] += position_df[name] * inv_mae
			position_current_df['Top Models Weighted Avg'] += position_current_df[name] * inv_mae
			total_inv_mae += inv_mae

			#calculate feature importance
			try:
				feature_importance = model.feature_importances_
				pd.DataFrame({'feature name': features, 'feature importance': feature_importance}).to_csv(name + ' ' + position_name + ' feature importance.csv', index=False)
			except AttributeError:
				pass

			#export partial dependence plot(only available for GBM implementation right now)
			if name is 'Top GBM':
				feature_importance = model.feature_importances_
				sorted_top10_idx = np.argsort(feature_importance)[:10]
				#feature_names = [features[i] for i in sorted_top5_idx]
				with warnings.catch_warnings(): #ignore warnings 
					warnings.simplefilter('ignore')
					fig, axs = plot_partial_dependence(model, X=x, features=sorted_top10_idx, feature_names=features, grid_resolution=10)
					#fig.show()
					fig.savefig(position_name + '_partial_dependence.png')
			
			#use for debugging weird Ridge Regression values
			#if 'Ridge' in name:
			#	print(model.coef_ )
			#	print(model.intercept_ )

			#update final model if this one is better
			if mae < best_mae:
				best_mae_col_name = name
	
			print('Model %s: %f MAE, %f R^2 in %f seconds' % (name, mae, r2, t1-t0))

		else:
			print('Model %s: in %f seconds' % (name, t1-t0))

	#calculate and evaluate weighted average of top level models
	position_df['Top Models Weighted Avg'] /= total_inv_mae
	position_current_df['Top Models Weighted Avg'] /= total_inv_mae
	mae = mean_absolute_error(y, position_df['Top Models Weighted Avg'])
	r2 = r2_score(y, position_df['Top Models Weighted Avg'])
	print('Weighted Avg of Top Models: %f MAE, %f R^2' % (mae, r2))

	#calculate min and max of top level models
	position_df['Top Models Minimum'] = position_df[[col for col in position_df.columns if 'Top' in col and '.' not in col]].min(axis=1)
	position_current_df['Top Models Minimum'] = position_current_df[[col for col in position_current_df.columns if 'Top' in col and '.' not in col]].min(axis=1)
	position_df['Top Models Maximum'] = position_df[[col for col in position_df.columns if 'Top' in col and '.' not in col and 'Min' not in col]].max(axis=1)
	position_current_df['Top Models Maximum'] = position_current_df[[col for col in position_current_df.columns if 'Top' in col and '.' not in col and 'Min' not in col]].max(axis=1)

	#update final model if this weighted average is better
	if mae < best_mae:
		best_mae_col_name = 'Top Models Weighted Avg'

	#make extra column for best performing model to use 
	position_df['Final Model'] = position_df[best_mae_col_name]
	position_current_df['Final Model'] = position_current_df[best_mae_col_name]

	#inverse transform points
	#position_df['Final Model'] = transformer.inverse_transform(position_df['Final Model'].values.reshape(-1, 1))
	#position_current_df['Final Model'] = transformer.inverse_transform(position_current_df['Final Model'].values.reshape(-1, 1))

	final_df = pd.concat([final_df, position_df], axis=0)
	final_current_df = pd.concat([final_current_df, position_current_df], axis=0)

	print(best_mae_col_name + ' is the Best Model for ' + position_name)

#write out best model's evaluative metrics
mae = mean_absolute_error(final_df[target], final_df['Final Model'])
r2 = r2_score(final_df[target], final_df['Final Model'])
print('The Best Model Overall: %f MAE, %f R^2' % (mae, r2))

#write to csv
final_df.to_csv(loc_train_output, sep=',', index=False)
final_current_df.to_csv(loc_current_output, sep=',', index=False)