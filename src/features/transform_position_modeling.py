import os
import sys
SRC_DIRECTORY = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(SRC_DIRECTORY))
os.chdir(SRC_DIRECTORY)
import pandas as pd
import numpy as np
import warnings
from src.utilities import data_utilities as utilities

def create_player_avgs(df):
	dict = {stat_name:{#TODO: try using standard deviation instead of quantiles
	'mean':'mean',
	'25%':(lambda x : x.quantile(0.25)),
	'75%':(lambda x : x.quantile(0.75))} for stat_name in utilities.clustering_stat_names()}
	dict['gls']['count'] = 'count'
	df = df.groupby(['player_id', 'season'], as_index=False).agg(dict)
	df.columns = ['_'.join(tup).rstrip('_') for tup in df.columns.values]
	df = df.rename({'gls_count': 'games'}, axis='columns')
	return df

def get_player_info(df):
	df_player_info = df[['player_name', 'player_id', 'position_id', 'season']].copy()
	df_player_info = df_player_info.drop_duplicates()
	return df_player_info

if __name__ == '__main__':
	warnings.filterwarnings('ignore', category=DeprecationWarning)
	warnings.filterwarnings('ignore', category=FutureWarning)
	df_historical = pd.read_csv(utilities.loc_historical_player_stats, sep=',', encoding='ISO-8859-1')

	df_historical['cl+blk+intc+br'] = df_historical['cl'] + df_historical['blk'] + df_historical['intc'] + df_historical['br']

	df_player_avgs = df_historical.pipe(create_player_avgs)
	df_player_info = df_historical.pipe(get_player_info)
	df_final = pd.merge(df_player_avgs, df_player_info, how='left', on=['player_id', 'season'])
	#df_final = df_final.dropna(axis=0, subset=['player_name'])
	df_final.to_csv(utilities.loc_clustering_data, sep=',', index=False)
