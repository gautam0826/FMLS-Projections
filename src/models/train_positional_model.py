import os
import sys
SRC_DIRECTORY = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(SRC_DIRECTORY))
os.chdir(SRC_DIRECTORY)
import numpy as np
import pandas as pd
from sklearn import metrics, svm, linear_model, naive_bayes, ensemble, discriminant_analysis, manifold, decomposition, preprocessing, cluster, mixture
import umap.umap_ as umap
import warnings
from itertools import chain
from time import time
from sklearn.exceptions import ConvergenceWarning
from src.utilities import data_utilities as utilities

MIN_GAMES = 5
N_COMPONENTS = 2

def manual_detailed_position(row, stats_dict):
	if row['position_id'] == 2:
		if (row['cl_mean'] > stats_dict['2_cl_mean'] or row['blk_mean'] > stats_dict['2_blk_mean']) or (row['cl_mean'] > stats_dict['2_cl_upper_quantile'] and
		row['blk_mean'] > stats_dict['2_blk_upper_quantile']):
			if ((row['crs_mean'] > stats_dict['2_crosses_mean']) and not (row['cl_mean'] > stats_dict['2_cl_upper_quantile']
			or row['blk_mean'] > stats_dict['2_blk_upper_quantile'])):
				return 'Full-back'
			elif (row['crs_mean'] > stats_dict['2_crosses_upper_quantile'] or row['kp_mean'] > stats_dict['2_kp_upper_quantile']):
				return 'Full-back'
			else:
				return 'Center-back'
		elif (row['crs_mean'] > stats_dict['2_crosses_upper_quantile'] or row['bc_mean'] > stats_dict['2_bc_upper_quantile'] or
		row['sh_mean'] > stats_dict['2_sh_upper_quantile'] or row['kp_mean'] > stats_dict['2_kp_upper_quantile']):
			return 'Wing-back'
		else:
			return 'Full-back'
	elif row['position_id'] == 3:
		if row['cl+blk+intc+br_mean'] > stats_dict['3_cbir_mean'] and row['sh_mean'] < stats_dict['3_sh_upper_quantile']:
			return 'Defensive Midfielder'
		if (row['gls_mean'] > stats_dict['3_gls_mean'] or row['kp_mean'] > stats_dict['3_kp_mean']):
			if (row['bc_mean'] > stats_dict['3_bc_upper_quantile'] or row['kp_mean'] > stats_dict['3_kp_upper_quantile']) and (row['pss_mean'] > stats_dict['3_pss_mean']):
				return 'Attacking Midfielder'
			else:
				return 'Attacking Winger'
		elif (row['crs_mean'] > stats_dict['3_crs_mean'] or
		row['br_mean'] < stats_dict['3_br_lower_quantile'] and row['kp_mean'] > stats_dict['3_kp_lower_quantile']):
			return 'Winger'
		else:
			return 'Rounded Midfielder'
	elif row['position_id'] == 4: #TODO: make Poacher broader to achieve?
		if row['aps_mean'] < stats_dict['4_aps_lower_quantile'] and row['sh_mean'] > stats_dict['4_sh_mean']:
			return 'Poacher'
		elif row['sh_mean'] > stats_dict['4_sh_upper_quantile'] and row['kp_mean'] > stats_dict['4_kp_upper_quantile'] and row['aps_mean'] > stats_dict['4_aps_lower_quantile']:
			return 'Elite Forward'
		elif row['sh_mean'] > stats_dict['4_sh_upper_quantile'] or row['gls_mean'] > stats_dict['4_gls_upper_quantile']:
			return 'High-volume Finisher'
		elif (row['kp_mean'] > stats_dict['4_kp_upper_quantile'] or row['br_mean'] > stats_dict['4_br_upper_quantile']
		or row['aps_mean'] > stats_dict['4_aps_upper_quantile']) and row['sh_mean'] < stats_dict['4_sh_upper_quantile']:
			return 'Second Striker'
		else:
			return 'Rounded Forward'
	else:
		return 'Goalkeeper'

#attacking mid even higher kp and maybe less def actions, higher passes and very high bcc
def predict_detailed_position_manual(df):
	stats_dict = {}
	stats_dict['2_cl_mean'] = df.loc[df['position_id'] == 2]['cl_mean'].mean()
	stats_dict['2_blk_mean'] = df.loc[df['position_id'] == 2]['blk_mean'].mean()
	stats_dict['2_cl_upper_quantile'] = df.loc[df['position_id'] == 2]['cl_mean'].quantile(0.75)
	stats_dict['2_blk_upper_quantile'] = df.loc[df['position_id'] == 2]['blk_mean'].quantile(0.75)
	stats_dict['2_crosses_mean'] = df.loc[df['position_id'] == 2]['crs_mean'].mean()
	stats_dict['2_crosses_upper_quantile'] = df.loc[df['position_id'] == 2]['crs_mean'].quantile(0.9)
	stats_dict['2_bc_upper_quantile'] = df.loc[df['position_id'] == 2]['bc_mean'].quantile(0.9)
	stats_dict['2_kp_upper_quantile'] = df.loc[df['position_id'] == 2]['kp_mean'].quantile(0.9)
	stats_dict['2_sh_upper_quantile'] = df.loc[df['position_id'] == 2]['sh_mean'].quantile(0.9)
	stats_dict['3_sh_upper_quantile'] = df.loc[df['position_id'] == 3]['sh_mean'].quantile(0.75)
	stats_dict['2_kp_lower_quantile'] = df.loc[df['position_id'] == 2]['kp_mean'].quantile(0.25)
	stats_dict['3_cbir_mean'] = df.loc[df['position_id'] == 3]['cl+blk+intc+br_mean'].mean()
	stats_dict['3_crosses_mean'] = df.loc[df['position_id'] == 3]['crs_mean'].mean()
	stats_dict['3_pss_mean'] = df.loc[df['position_id'] == 3]['pss_mean'].mean()
	stats_dict['3_bc_upper_quantile'] = df.loc[df['position_id'] == 3]['bc_mean'].quantile(0.75)
	stats_dict['3_gls_mean'] = df.loc[df['position_id'] == 3]['gls_mean'].mean()
	stats_dict['3_kp_mean'] = df.loc[df['position_id'] == 3]['kp_mean'].mean()
	stats_dict['3_crs_mean'] = df.loc[df['position_id'] == 3]['crs_mean'].mean()
	stats_dict['3_kp_upper_quantile'] = df.loc[df['position_id'] == 3]['kp_mean'].quantile(0.75)
	stats_dict['3_kp_lower_quantile'] = df.loc[df['position_id'] == 3]['kp_mean'].quantile(0.25)
	stats_dict['3_br_lower_quantile'] = df.loc[df['position_id'] == 3]['br_mean'].quantile(0.25)
	stats_dict['4_kp_upper_quantile'] = df.loc[df['position_id'] == 4]['kp_mean'].quantile(0.75)
	stats_dict['4_sh_upper_quantile'] = df.loc[df['position_id'] == 4]['sh_mean'].quantile(0.75)
	stats_dict['4_sh_mean'] = df.loc[df['position_id'] == 4]['sh_mean'].mean()
	stats_dict['4_aps_upper_quantile'] = df.loc[df['position_id'] == 4]['aps_mean'].quantile(0.75)
	stats_dict['4_aps_lower_quantile'] = df.loc[df['position_id'] == 4]['aps_mean'].quantile(0.25)
	stats_dict['4_br_upper_quantile'] = df.loc[df['position_id'] == 4]['br_mean'].quantile(0.75)
	stats_dict['4_gls_upper_quantile'] = df.loc[df['position_id'] == 4]['gls_mean'].quantile(0.9)

	df['detailed_position_manual'] = df.apply(manual_detailed_position, stats_dict=stats_dict, axis=1)
	return df

def process_model_predictions(df, target):
	position_columns = df.columns
	df[target + '_model'] = df[position_columns].idxmax(axis=1)
	df[target + '_max_prob'] = df[position_columns].max(axis=1)
	return df

def predict_detailed_goalkeeper_model(df_position, dimensionality_reducers):
	for name, reducer in dimensionality_reducers:
		for n in range(1, N_COMPONENTS + 1):
			df_position[name + '_attack_component_' + str(n)] = 0
			df_position[name + '_defense_component_' + str(n)] = 0
	df_position['detailed_position_model'] = 'Goalkeeper'
	df_position['max_prob'] = 1
	return df_position

def predict_detailed_field_position_model(df_position, dimensionality_reducers, attacking_features, defending_features):
	position_dict = {
		'attack': attacking_features,
		'defense': defending_features
	}
	df_low_gametime = df_position.loc[df_position['games'] < MIN_GAMES].copy().reset_index(drop=True)
	df = df_position.loc[df_position['games'] >= MIN_GAMES].copy().reset_index(drop=True)

	y = df['detailed_position_manual']
	y_encoded = preprocessing.LabelEncoder().fit_transform(y)

	for (component_name, component_features) in position_dict.items():
		X_component = df[component_features]
		X_component_low_gametime = df_low_gametime[component_features]

		for name, reducer in dimensionality_reducers:
			reducer.fit(X_component, y_encoded)
			df_component_transformed = pd.DataFrame(reducer.transform(X_component),
				columns=[name + '_' + component_name + '_component_' + str(n) for n in range(1, N_COMPONENTS+1)])
			df_component_transformed_low_gametime = pd.DataFrame(reducer.transform(X_component_low_gametime),
				columns=[name + '_' + component_name + '_component_' + str(n) for n in range(1, N_COMPONENTS+1)])
			df = pd.concat([df, df_component_transformed], axis=1)
			df_low_gametime = pd.concat([df_low_gametime, df_component_transformed_low_gametime], axis=1)

	model_features = [prefix + str(n) for n in range(1, N_COMPONENTS + 1) for prefix in ('LDA_attack_component_', 'LDA_defense_component_')]
	X = df[model_features]
	X_low_gametime = df_low_gametime[model_features]

	model = svm.SVC(probability=True, gamma='auto')
	model = linear_model.LogisticRegression()
	model.fit(X, y)

	df_predictions = pd.DataFrame(model.predict_proba(X), columns=model.classes_)
	df_predictions = df_predictions.pipe(process_model_predictions, target='detailed_position')
	df_predictions_low_gametime = pd.DataFrame(model.predict_proba(X_low_gametime), columns=model.classes_)
	df_predictions_low_gametime = df_predictions_low_gametime.pipe(process_model_predictions, target='detailed_position')

	df = pd.concat([df, df_predictions], axis=1)
	df_low_gametime = pd.concat([df_low_gametime, df_predictions_low_gametime], axis=1)
	df_final = pd.concat([df, df_low_gametime], axis=0, ignore_index=True, sort=True)

	return df_final

def predict_detailed_position_model(df):
	#set aside unused columns when getting features
	unused_cols = ['player_name', 'player_id', 'position_id', 'season', 'games', 'detailed_position_manual']
	unused_cols.extend([col for col in df.columns if '+' not in col])
	attacking_features = list(chain.from_iterable((feature + '_mean', feature + '_25%', feature + '_75%') for feature in utilities.attacking_stat_names()))
	defending_features = list(chain.from_iterable((feature + '_mean', feature + '_25%', feature + '_75%') for feature in utilities.defending_stat_names()))

	dimensionality_reducers = [
		('PCA', decomposition.PCA(n_components=N_COMPONENTS)),
		('UMAP', umap.UMAP(n_components=N_COMPONENTS)),
		('LDA', discriminant_analysis.LinearDiscriminantAnalysis(n_components=N_COMPONENTS)),
	]

	df_final = pd.DataFrame()
	extra = 1 #amount to shift over clusters for each position so different positions have different cluster spaces
	for i in range(1, 5):
		df_position = df.loc[df['position_id'] == i].copy()
		if i == 1:
			df_position = predict_detailed_goalkeeper_model(df_position, dimensionality_reducers)
		else:
			df_position = predict_detailed_field_position_model(df_position, dimensionality_reducers, attacking_features, defending_features)
		df_final = pd.concat([df_final, df_position], axis=0, ignore_index=True, sort=True)

	columns_to_fill = {col:0 for col in df_final.columns if 'detailed' not in col}
	df_final = df_final.fillna(columns_to_fill)
	return df_final

def team_(df):
	df_top_shooters = df.groupby(['player_id']).head(2).reset_index(drop=True)
	df_top_creators = df.groupby(['player_id']).head(2).reset_index(drop=True)
	df_main_defenders = df.groupby(['player_id']).head(1).reset_index(drop=True)

def predict_out_of_position_model(df):
	attacking_features = list(chain.from_iterable((feature + '_mean', feature + '_25%', feature + '_75%') for feature in utilities.attacking_stat_names()))
	defending_features = list(chain.from_iterable((feature + '_mean', feature + '_25%', feature + '_75%') for feature in utilities.defending_stat_names()))
	features = attacking_features + defending_features

	X = df[features]
	y = df['position_id']

	model = svm.SVC(probability=True, gamma='auto')
	model = linear_model.LogisticRegression()
	model.fit(X, y)

	df_predictions = pd.DataFrame(model.predict_proba(X), columns=model.classes_)
	df_predictions = df_predictions.pipe(process_model_predictions, target='position_id')
	df = pd.concat([df, df_predictions], axis=1)
	return df


if __name__ == '__main__':
	pd.options.mode.chained_assignment = None  #gets rid of SettingWithCopyWarnings
	warnings.filterwarnings('ignore', category=DeprecationWarning)
	warnings.filterwarnings('ignore', category=ConvergenceWarning)

	df = pd.read_csv(utilities.loc_clustering_data, sep=',', encoding='ISO-8859-1')
	df = df.fillna(0) #nas probably came from divide by 0 errors during preprocessing
	df = df.pipe(predict_detailed_position_manual)
	df = df.pipe(predict_detailed_position_model)
	df = df.pipe(predict_out_of_position_model)

	#write to csv
	df.to_csv(utilities.loc_clustering_output, sep=',', index=False)
