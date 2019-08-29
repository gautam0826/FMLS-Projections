import os
import sys
SRC_DIRECTORY = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(SRC_DIRECTORY))
os.chdir(SRC_DIRECTORY)
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
from src.utilities import data_utilities as utilities

def fill_unknown_advanced_position_func(row):
	if not pd.isnull(row['detailed_position_manual']):
		return row['detailed_position_manual']
	elif row['position_id'] == 2:
		return 'Full-back'
	elif row['position_id'] == 3:
		return 'Rounded Midfielder'
	elif row['position_id'] == 4:
		return 'Rounded Forward'
	else:
		return 'Goalkeeper'

def fill_unknown_model_position_func(row):
	if not pd.isnull(row['position_id_model']):
		return row['position_id_model']
	else:
		return row['position_id']

def add_clustering(df):
	df_clustering = pd.read_csv(utilities.loc_clustering_output, sep=',', encoding='ISO-8859-1')
	df_final = pd.merge(df, df_clustering[['player_id', 'season', 'detailed_position_manual', 'position_id_model']], how='left', on=['player_id', 'season'])
	df_final['detailed_position_manual'] = df_final.apply(fill_unknown_advanced_position_func, axis=1)
	df_final['position_id_model'] = df_final.apply(fill_unknown_model_position_func, axis=1)
	return df_final

def add_lagging_column(df, groupby_list, feature_name, lag_amount, new_col_suffix):
	new_col_name = feature_name + '_lag_' + str(lag_amount) + '_' + new_col_suffix
	df[new_col_name] = (df.groupby(groupby_list)[feature_name].shift(lag_amount))
	df[new_col_name] = (df.groupby(groupby_list)[new_col_name].fillna(method='ffill'))
	return df

def add_lagging_features(df, groupby_list, feature_names, lag_amounts, new_col_suffix):
	for feature_name in tqdm(feature_names):
		for lag_amount in lag_amounts:
			df = df.pipe(add_lagging_column, groupby_list, feature_name, lag_amount, new_col_suffix)
	return df

def add_player_lagging_features(df):
	feature_names = utilities.important_feature_names() + ['home']
	groupby_tuples = [
		(['player_id'], list(range(1, 8, 1)), 'player'),
		(['opponent', 'detailed_position_manual'], list(range(1, 7, 1)), 'opp'),
	]
	for groupby_list, lag_amounts, new_col_suffix in groupby_tuples:
		df = df.pipe(add_lagging_features, groupby_list, feature_names, lag_amounts, new_col_suffix)
	return df

def add_team_lagging_features(df):
	feature_names = utilities.all_feature_names()
	feature_names.remove('mins')
	feature_names.remove('cs')
	feature_names.remove('gc')
	team_feature_names = ['team_' + feature for feature in feature_names] + ['opp_' + feature for feature in feature_names] + ['home']
	groupby_tuples = [
		(['team'], list(range(1, 7, 1)), 'team_total'),
		(['opponent'], list(range(1, 7, 1)), 'opp_total'),
	]
	for groupby_list, lag_amounts, new_col_suffix in groupby_tuples:
		df = df.pipe(add_lagging_features, groupby_list, team_feature_names, lag_amounts, new_col_suffix)
	return df

def add_team_opponent_features(df):
	df_team_stats = pd.read_csv(utilities.loc_historical_team_stats, sep=',', encoding='ISO-8859-1')
	df_team_stats.pipe(add_round_match_count)
	feature_names = utilities.all_feature_names()
	feature_names.remove('mins')
	feature_names.remove('cs')
	feature_names.remove('gc')
	df_team_stats = df_team_stats.pipe(add_team_lagging_features)
	remove_cols = ['team_' + feature for feature in feature_names]
	remove_cols.extend(['opp_' + feature for feature in feature_names])
	remove_cols.extend(['home'])
	df_team_stats = df_team_stats.drop(remove_cols, axis=1, errors='ignore')
	df = pd.merge(df, df_team_stats, how='left', on=['event_id', 'team', 'opponent', 'season', 'round'])
	return df

def one_hot_encoding(df):
	categorical_features = ['detailed_position_manual']
	df = pd.get_dummies(df, columns=categorical_features)
	return df

def filter_players_by_mins(df):
	df = df.loc[df['mins'] >= 45].copy()
	return df

#should be used to detect international breaks
def add_round_match_count(df):
	df['round_match_count'] = df.groupby(['season', 'round'], as_index=False)['event_id'].transform('count')/2
	return df

if __name__ == '__main__':
	warnings.filterwarnings('ignore', category=DeprecationWarning)

	df_historical = pd.read_csv(utilities.loc_historical_player_stats, sep=',', encoding='ISO-8859-1')
	df_current = pd.read_csv(utilities.loc_current_gameweek, sep=',', encoding='ISO-8859-1')

	df_historical = df_historical.pipe(filter_players_by_mins)
	df_final = pd.concat([df_historical, df_current], axis=0, ignore_index=False, sort=True).copy()
	df_final = df_final.reset_index()

	df_final = df_final.pipe(add_clustering)
	df_final = df_final.pipe(add_player_lagging_features)
	df_final = df_final.pipe(add_team_opponent_features)
	df_final = df_final.pipe(one_hot_encoding)

	remove_cols = utilities.all_stat_names()
	remove_cols.extend(['points', 'att_points', 'def_points','att_bps', 'def_bps', 'pas_bps', 'index_x', 'index_y'])
	df_final = df_final.drop(remove_cols, axis=1, errors='ignore')

	df_final_historical = df_final.loc[df_final['dataset'] != 'live'].copy()
	df_final_current = df_final.loc[df_final['dataset'] == 'live'].copy()

	df_final_historical = df_final_historical.dropna(axis=1, how='all')
	df_final_current = df_final_current.dropna(axis=1, how='all')

	df_final_historical = df_final_historical.fillna(-1)
	df_final_current = df_final_current.fillna(-1)

	df_final_train = df_final_historical.loc[(df_final_historical['dataset'] == 'training') | (df_final_historical['dataset'] == 'validation')].copy()
	df_final_test = df_final_historical.loc[(df_final_historical['dataset'] != 'training') & (df_final_historical['dataset'] != 'validation')].copy()

	df_final_train.to_csv(utilities.loc_train_nn_data, sep=',', index=False)
	df_final_test.to_csv(utilities.loc_test_nn_data, sep=',', index=False)
	df_final_current.to_csv(utilities.loc_current_nn_data, sep=',', index=False)
