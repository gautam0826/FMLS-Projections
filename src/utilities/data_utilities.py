import os
import math
import pandas as pd

subdir_seventeen = '2017data'
subdir_eighteen = '2018data'
subdir_nineteen = '2019data'

path_to_processed = os.path.join('..', 'data', 'processed')
path_to_interim = os.path.join('..', 'data', 'interim')
path_to_raw = os.path.join('..', 'data', 'raw')
path_to_models = os.path.join('..', 'models')

file_prefix = '2019_'

loc_name_matching = os.path.join(path_to_processed, file_prefix + 'player_ids.csv')
loc_historical_stageing = os.path.join(path_to_interim, file_prefix + 'staged_data.csv')
loc_historical_player_stats = os.path.join(path_to_processed, file_prefix + 'historical_player_stats.csv')
loc_historical_team_stats = os.path.join(path_to_processed, file_prefix + 'historical_team_stats.csv')
loc_current_gameweek = os.path.join(path_to_processed, file_prefix + 'current_gameweek.csv')
loc_historical_player_anomalies = os.path.join(path_to_processed, file_prefix + 'historical_player_anomalies.csv')
loc_train_nn_data = os.path.join(path_to_processed, file_prefix + 'training_nn_data.csv')
loc_test_nn_data = os.path.join(path_to_processed, file_prefix + 'testing_nn_data.csv')
loc_current_nn_data = os.path.join(path_to_processed, file_prefix + 'current_nn_data.csv')
loc_test_nn_predictions = os.path.join(path_to_processed, file_prefix + 'testing_nn_predictions.csv')
loc_current_nn_predictions = os.path.join(path_to_processed, file_prefix + 'current_nn_predictions.csv')
loc_train_gbm_data = os.path.join(path_to_processed, file_prefix + 'training_gbm_data.csv')
loc_test_gbm_data = os.path.join(path_to_processed, file_prefix + 'testing_gbm_data.csv')
loc_current_gbm_data = os.path.join(path_to_processed, file_prefix + 'current_gbm_data.csv') #add shap validation testing and filters
loc_test_gbm_predictions = os.path.join(path_to_processed, file_prefix + 'testing_gbm_predictions.csv')
loc_current_gbm_predictions = os.path.join(path_to_processed, file_prefix + 'current_gbm_predictions.csv')
loc_clustering_data = os.path.join(path_to_processed, file_prefix + 'clustering_data.csv')
loc_clustering_output = os.path.join(path_to_processed, file_prefix + 'clustering_output.csv')

_att_bps_dict = {'sh':(1/4), 'crs':(1/3), 'kp':(1/3), 'bc':1, 'wf':(1/4), 'oga':1, 'pe':2}
_def_bps_dict = {'cl':(1/4), 'blk':(1/2), 'intc':(1/4), 'tck':(1/4), 'br':(1/6), 'sv':(1/3)}
_1_2_extra_att_pts_dict = {'gls':6, 'ass':3}
_1_2_extra_def_pts_dict = {'cs':5, 'gc':(-1/2)}
_3_extra_att_pts_dict = {'gls':5, 'ass':3}
_3_extra_def_pts_dict = {'cs':1}
_4_extra_att_pts_dict = {'gls':5, 'ass':3}
_4_extra_def_pts_dict = {}
_extra_adj_pts_dict = {'yc':-1, 'mins':(1/60)}
_extra_real_pts_dict = {'elg':-1, 'rc':-3, 'og':-2, 'ps':5, 'pm':-2}

def passing_stat_names():
	stat_names = ['pss', 'aps', 'pcp']
	return stat_names

def attacking_bonus_stat_names():
	return list(_att_bps_dict.keys())

def defending_bonus_stat_names():
	return list(_def_bps_dict.keys())

def attacking_stat_names():
	return attacking_bonus_stat_names() + list(_1_2_extra_att_pts_dict.keys())

def defending_stat_names():
	return defending_bonus_stat_names() + list(_1_2_extra_def_pts_dict.keys())

def important_stat_names():
	return attacking_stat_names() + defending_stat_names() + passing_stat_names() + list(_extra_adj_pts_dict.keys())

def all_stat_names():
	return important_stat_names() + list(_extra_real_pts_dict.keys())

def clustering_stat_names():
	return important_stat_names() + ['cl+blk+intc+br']

def minimum_important_feature_names():
	return ['adjusted_points', 'att_bps', 'def_bps', 'att_points', 'def_points', 'pas_bps', 'total_bps']

def important_feature_names():
	return important_stat_names() + minimum_important_feature_names()

def all_feature_names():
	return all_stat_names() + minimum_important_feature_names()

def fantasy_score(stats_dict):
	position_id = stats_dict['position_id']
	if position_id == 1 or position_id == 2:
		_extra_att_pts_dict = _1_2_extra_att_pts_dict
		_extra_def_pts_dict = _1_2_extra_def_pts_dict
	elif position_id == 3:
		_extra_att_pts_dict = _3_extra_att_pts_dict
		_extra_def_pts_dict = _3_extra_def_pts_dict
	else:
		_extra_att_pts_dict = _4_extra_att_pts_dict
		_extra_def_pts_dict = _4_extra_def_pts_dict
	_extra_adj_pts_dict = {'yc':-1, 'mins':(1/60)}
	_extra_real_pts_dict = {'elg':-1, 'rc':-3, 'og':-2, 'ps':5, 'pm':-2}

	att_bps = sum([math.floor(stats_dict[stat_type] * multiplier) for (stat_type, multiplier) in _att_bps_dict.items()])
	def_bps = sum([math.floor(stats_dict[stat_type] * multiplier) for (stat_type, multiplier) in _def_bps_dict.items()])
	pas_bps = (stats_dict['pss'] // 35) if (stats_dict['pcp'] >= .85) else 0
	total_bps = att_bps + def_bps + pas_bps
	extra_att_pts = sum([(stats_dict[stat_type] * multiplier) for (stat_type, multiplier) in _extra_att_pts_dict.items()])
	extra_def_pts = sum([math.ceil(stats_dict[stat_type] * multiplier) for (stat_type, multiplier) in _extra_def_pts_dict.items()])
	extra_adj_pts = sum([math.floor(stats_dict[stat_type] * multiplier) for (stat_type, multiplier) in _extra_adj_pts_dict.items()]) + int(stats_dict['mins'] > 0)
	extra_real_pts = sum([(stats_dict[stat_type] * multiplier) for (stat_type, multiplier) in _extra_real_pts_dict.items()])
	att_pts = att_bps + extra_att_pts
	def_pts = def_bps + extra_def_pts
	adj_pts = att_pts + def_pts + pas_bps + extra_adj_pts
	real_pts = adj_pts + extra_real_pts
	stats_dict.update({'adjusted_points': adj_pts, 'points':real_pts, 'att_points':att_pts, 'def_points':def_pts, 'att_bps':att_bps, 'def_bps':def_bps, 'pas_bps':pas_bps, 'total_bps':total_bps})
	return stats_dict
