import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import inspect
import warnings
import time
import mord
from gplearn import genetic
from pyearth import Earth
from pygam import LinearGAM
from sklearn.model_selection import train_test_split
from sklearn import metrics, svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process, neural_network
#from sklearn.base import clone
#from sklearn.ensemble.partial_dependence import plot_partial_dependence
#from sklearn.ensemble.partial_dependence import partial_dependence
#from sklearn.preprocessing import QuantileTransformer
from sklearn.exceptions import ConvergenceWarning
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from nonconformist.cp import IcpRegressor
from nonconformist.nc import NcFactory
pd.options.mode.chained_assignment = None  #gets rid of SettingWithCopyWarnings
warnings.filterwarnings("ignore", category=ConvergenceWarning) #get rid of Convergence Warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) #get rid of Deprecation Warnings

loc_train = '2018_input_data.csv'
loc_current= '2018_current_data.csv'
loc_test_output = '2018_test_predictions.csv'
loc_current_output = '2018_current_predictions.csv'
loc_model_info = 'model_performance.csv'

df = pd.read_csv(os.path.join('..', '..', 'data', 'processed', loc_train), sep=',', encoding='ISO-8859-1')
df_current = pd.read_csv(os.path.join('..', '..', 'data', 'processed', loc_current), sep=',', encoding='ISO-8859-1')

#set aside unused columns when getting features
unused_cols = ['index', 'adjusted points', 'att bps', 'def bps', 'pas bps', 'cost', 'event id', 'mins', 'player id', 'player name', 'points', 'round', 'team id', 'transfers in', 'transfers out', 'seven plus', 'ten plus',  'adj points/transfers in', 'adj points/last 5', 'adj points/h adj last 3', 'position id_1', 'position id_2', 'position id_3', 'position id_4', 'season', 'importance']
#comment out line below if you want to use binary opponent variables, for the upcoming season I will not because of major changes to team strengths
unused_cols.extend([col for col in df.columns if 'opponent' in col or 'position' in col])
features = [col for col in df.columns if col not in unused_cols]
features_alt = [col for col in features if 'opp' not in col]
features_alt.extend([col for col in df.columns if 'opponent' in col])
target = 'adjusted points'
target_7 = 'seven plus'
target_10 = 'ten plus'
importance = 'importance'

num_testing_weeks = 5
current_week = df_current['round'].max()
df_train = df.loc[(df['season'] == 2017) | (df['round'] < current_week - num_testing_weeks)]
df_test = df.loc[(df['season'] == 2018) & (df['round'] >= current_week - num_testing_weeks)]

models = [
('Ridge', linear_model.Ridge(alpha=1)),
('Huber', linear_model.HuberRegressor(epsilon=1.5)),
('RANSAC', linear_model.RANSACRegressor()),
('SVM', svm.LinearSVR(C=0.01, dual=True, epsilon=1.0, loss='epsilon_insensitive', tol=1e-05)),
('ET', ensemble.ExtraTreesRegressor(bootstrap=False, min_samples_leaf=10, min_samples_split=20, n_estimators=100)),
('GBM', ensemble.GradientBoostingRegressor()),
('KNN', neighbors.KNeighborsRegressor(n_neighbors=10)),
#('GAM', LinearGAM()),
#('Genetic', genetic.SymbolicRegressor(population_size=200)),
('MARS', Earth()),]

models_with_samp_weights = [
('Ridge weighted', linear_model.Ridge(alpha=1)),
('Huber weighted', linear_model.HuberRegressor(epsilon=1.5)),
('RANSAC weighted', linear_model.RANSACRegressor()),
('SVM weighted', svm.LinearSVR(C=0.01, dual=True, epsilon=1.0, loss='epsilon_insensitive', tol=1e-05)),
('ET weighted', ensemble.ExtraTreesRegressor(bootstrap=False, min_samples_leaf=10, min_samples_split=20, n_estimators=100)),
('GBM weighted', ensemble.GradientBoostingRegressor()),
#('GAM weighted', LinearGAM()),
#('Genetic weighted', genetic.SymbolicRegressor(population_size=200)),
('MARS weighted', Earth()),]

classifiers = [
('Logistic1', linear_model.LogisticRegression(C=1)),
('Logistic2', linear_model.LogisticRegression(C=100)),
('NB', naive_bayes.GaussianNB()),
('GBM', ensemble.GradientBoostingClassifier()),
('RF', ensemble.RandomForestClassifier(n_estimators=15, min_samples_split=5)),]

samplers = [
('SMOTETOMEK', SMOTETomek()),
('SMOTEENN', SMOTEENN()),]

model_performance = []

def score(df_test, column, position_name=None):
	position_counts = {'position id_1.0':3, 'position id_2.0':6, 'position id_3.0':6, 'position id_4.0':4}
	sum = 0
	if (position_name is None):
		for round in df_test['round'].unique():
			df_round = df_test.loc[df_test['round'] == round].sort_values(column, ascending=False)
			df_test.loc[df_test['round'] == round]
			for position_name, count_players in position_counts.items():
				sum = sum + df_round.loc[df_round[position_name] == 1]['adjusted points'].head(position_counts[position_name]).sum()
		return (sum / (num_testing_weeks * 19))
	else:
		for round in df_test['round'].unique():
			df_round = df_test.loc[df_test['round'] == round].sort_values(column, ascending=False)
			sum += df_round.loc[df_round[position_name] == 1]['adjusted points'].head(position_counts[position_name]).sum()
		return (sum / (num_testing_weeks * position_counts[position_name]))

#select only the position being modeled
for position_name in ['position id_1.0', 'position id_2.0', 'position id_3.0', 'position id_4.0']:
	X = df.loc[df[position_name] == 1][features]
	X_train = df_train.loc[df_train[position_name] == 1][features]
	X_test = df_test.loc[df_test[position_name] == 1][features]
	X_current = df_current.loc[df_current[position_name] == 1][features]
	y = df.loc[df[position_name] == 1][target]
	y_train = df_train.loc[df_train[position_name] == 1][target]
	y_test = df_test.loc[df_test[position_name] == 1][target]
	sample_weight = df.loc[df[position_name] == 1][importance]
	sample_weight_train = df_train.loc[df_train[position_name] == 1][importance]

	X_alt = df.loc[df[position_name] == 1][features_alt]
	X_train_alt = df_train.loc[df_train[position_name] == 1][features_alt]
	X_test_alt = df_test.loc[df_test[position_name] == 1][features_alt]
	X_current_alt = df_current.loc[df_current[position_name] == 1][features_alt]

	y_7 = df.loc[df[position_name] == 1][target_7]
	y_train_7 = df_train.loc[df_train[position_name] == 1][target_7]
	y_10 = df.loc[df[position_name] == 1][target_10]
	y_train_10 = df_train.loc[df_train[position_name] == 1][target_10]

	X_train_conf, X_calibrate, y_train_conf, y_calibrate = train_test_split(X_train, y_train, test_size=0.2)

	for model_name, model in models:
		model.fit(X_train, y_train)
		df_test.loc[df_test[position_name] == 1, model_name + ' pred'] = model.predict(X_test)
		time1 = time.time()
		model.fit(X, y)
		df_current.loc[df_current[position_name] == 1, model_name + ' pred'] = model.predict(X_current)
		time2 = time.time()
		model_performance.append({'model':model_name, 'score':score(df_test, column=model_name + ' pred', position_name=position_name), 'r^2':metrics.r2_score(df_test[df_test[position_name] == 1]['adjusted points'], df_test[df_test[position_name] == 1][model_name + ' pred']), 'position':position_name, 'time':(time2-time1)})
		print('done doing model ' + model_name + ' for ' + position_name)

		#testing prediction intervals
		#if 'GAM' not in model_name:
		#	nc = NcFactory.create_nc(model)
		#	icp = IcpRegressor(nc)
		#	icp.fit(X_train_conf, y_train_conf)
		#	icp.calibrate(X_calibrate, y_calibrate)
		#	bounds = (icp.predict(X_test.values, significance=0.25))
		#	df_test.loc[df_test[position_name] == 1, model_name + ' lower'] = bounds[:,0]
		#	df_test.loc[df_test[position_name] == 1, model_name + ' upper'] = bounds[:,1]
		#	icp = IcpRegressor(nc)
		#	icp.fit(X_train, y_train)
		#	icp.calibrate(X_test, y_test)
		#	bounds = (icp.predict(X_current.values, significance=0.25))
		#	df_current.loc[df_current[position_name] == 1, model_name + ' lower'] = bounds[:,0]
		#	df_current.loc[df_current[position_name] == 1, model_name + ' upper'] = bounds[:,1]
		#	model_performance.append({'model':model_name + ' upper', 'score':score(df_test, column=model_name + ' upper', position_name=position_name), 'position':position_name})
		#	model_performance.append({'model':model_name + ' lower', 'score':score(df_test, column=model_name + ' lower', position_name=position_name), 'position':position_name})


	for model_name, model in models_with_samp_weights:
		if 'GAM' not in model_name:
			model.fit(X_train_alt, y_train, sample_weight=sample_weight_train)
		else:
			model.fit(X_train_alt, y_train, weights=sample_weight_train)
		df_test.loc[df_test[position_name] == 1, model_name + ' pred'] = model.predict(X_test_alt)
		time1 = time.time()
		if 'GAM' not in model_name:
			model.fit(X_alt, y, sample_weight=sample_weight)
		else:
			model.fit(X_alt, y, weights=sample_weight)
		df_current.loc[df_current[position_name] == 1, model_name + ' pred'] = model.predict(X_current_alt)
		time2= time.time()
		model_performance.append({'model':model_name, 'score':score(df_test, column=model_name + ' pred', position_name=position_name), 'r^2':metrics.r2_score(df_test[df_test[position_name] == 1]['adjusted points'], df_test[df_test[position_name] == 1][model_name + ' pred']), 'position':position_name, 'time':(time2-time1)})
		print('done doing model ' + model_name + ' for ' + position_name)

	'''for model_name, model in classifiers:
		model_name += ' 7+'
		model.fit(X_train, y_train_7)
		df_test.loc[df_test[position_name] == 1, model_name] = model.predict_proba(X_test)[:,1]
		model_performance.append({'model':model_name, 'score':score(df_test, column=model_name, position_name=position_name), 'position':position_name})
		model.fit(X, y_7)
		df_current.loc[df_current[position_name] == 1, model_name] = model.predict_proba(X_current)[:,1]
		for sampler_name, sampler in samplers:
			X_train_resampled, y_train_7_resampled = sampler.fit_sample(X_train, y_train_7)
			X_resampled, y_7_resampled = sampler.fit_sample(X, y_7)
			model.fit(X_train_resampled, y_train_7_resampled)
			df_test.loc[df_test[position_name] == 1, model_name + ' ' + sampler_name] = model.predict_proba(X_test)[:,1]
			model_performance.append({'model':model_name + ' ' + sampler_name, 'score':score(df_test, column=model_name + ' ' + sampler_name, position_name=position_name), 'position':position_name})
			model.fit(X_resampled, y_7_resampled)
			df_current.loc[df_current[position_name] == 1, model_name + ' ' + sampler_name] = model.predict_proba(X_current)[:,1]
			print('done doing model ' + model_name + ' ' + sampler_name + ' for ' + position_name)
		print('done doing model ' + model_name + ' for ' + position_name)

	for model_name, model in classifiers:
		model_name += ' 10+'
		model.fit(X_train, y_train_10)
		df_test.loc[df_test[position_name] == 1, model_name] = model.predict_proba(X_test)[:,1]
		model_performance.append({'model':model_name, 'score':score(df_test, column=model_name, position_name=position_name), 'position':position_name})
		model.fit(X, y_10)
		df_current.loc[df_current[position_name] == 1, model_name] = model.predict_proba(X_current)[:,1]
		for sampler_name, sampler in samplers:
			X_train_resampled, y_train_10_resampled = sampler.fit_sample(X_train, y_train_10)
			X_resampled, y_10_resampled = sampler.fit_sample(X, y_10)
			model.fit(X_train_resampled, y_train_10_resampled)
			df_test.loc[df_test[position_name] == 1, model_name + ' ' + sampler_name] = model.predict_proba(X_test)[:,1]
			model_performance.append({'model':model_name + ' ' + sampler_name, 'score':score(df_test, column=model_name + ' ' + sampler_name, position_name=position_name), 'position':position_name})
			model.fit(X_resampled, y_10_resampled)
			df_current.loc[df_current[position_name] == 1, model_name + ' ' + sampler_name] = model.predict_proba(X_current)[:,1]
			print('done doing model ' + model_name + ' ' + sampler_name + ' for ' + position_name)
		print('done doing model ' + model_name + ' for ' + position_name)'''

#write to csv
df_model_info = pd.DataFrame(model_performance)
df_test.to_csv(os.path.join('..', '..', 'data', 'processed', loc_test_output), sep=',', index=False)
df_current.to_csv(os.path.join('..', '..', 'data', 'processed', loc_current_output), sep=',', index=False)
df_model_info.to_csv(os.path.join('..', '..', 'data', 'processed', loc_model_info), sep=',', index=False)