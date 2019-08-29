import os
import sys
SRC_DIRECTORY = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(SRC_DIRECTORY))
os.chdir(SRC_DIRECTORY)
import json
import unidecode
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
import category_encoders as encoders
import warnings
from sklearn import model_selection
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from src.utilities import data_utilities as utilities

NUM_TESTING_ROUNDS = 5

def get_real_team(df):
	team_list = []
	for index, row in df.iterrows():
		team = row['team']
		opponent = row['opponent']
		if team in team_list and opponent not in team_list:
			return team
		if opponent in team_list and team not in team_list:
			return opponent
		team_list.append(team)
		team_list.append(opponent)
	return 'unfound'

def fill_correct_team(row, real_team):
	if row['team'] != real_team:
		row['opponent'] = row['team']
		row['team'] = real_team
		row['home'] = 1
	else:
		row['home'] = 0
	return row

def fix_player_home(df):
	df_final = df.loc[df['home'] != 2].copy()
	df_player_transfered = df.loc[df['home'] == 2]
	df_player_counts = df_player_transfered.groupby(['player_id', 'season']).size().reset_index(name='count')
	df_player_transfered = pd.merge(df_player_transfered, df_player_counts, how='left', on=['player_id', 'season'])
	df_temp = df_player_transfered.loc[df_player_transfered['count'] == 1]
	df_temp.to_csv(utilities.loc_historical_player_anomalies, sep=',', index=False)
	df_player_trans_final = pd.DataFrame()
	df_player_transfered = df_player_transfered.loc[df_player_transfered['count'] >= 2]
	for season in df_player_transfered['season'].unique():
		for player_id in df_player_transfered.loc[df_player_transfered['season'] == season]['player_id'].unique():
			df_subset = df.loc[(df['season'] == season) & (df['player_id'] == player_id)]
			min_index = ((df['season'] == season) & (df['player_id'] == player_id) & (df['home'].values == 2)).idxmax() #fix to add it to min index
			#df.loc[(df['season'] == season) & (df['player_id'] == player_id) & (df.index <= min_index), 'home'] = 2
			#df_subset = df.loc[(df['season'] == season) & (df['player_id'] == player_id) & (df.index <= min_index)]
			real_team = get_real_team(df_subset)

			if real_team != 'unfound':
				df_subset = df_subset.apply(fill_correct_team, args=(real_team,), axis=1)
				df_player_trans_final = pd.concat([df_player_trans_final, df_subset], axis=0, sort=True)
	#df_player_trans_final.to_csv('a.csv')
	df_final = pd.concat([df_final, df_player_trans_final])
	return df_final

def create_team_stats_df(df_historical, df_opponent):
	agg_dict = {stat_name:'sum' for stat_name in list(set(utilities.all_feature_names()) - set(['pcp', 'cs', 'gc', 'gls', 'mins']))}
	agg_dict['gc'] = 'max'
	agg_dict['home'] = 'max'
	agg_dict['round'] = 'max'
	df_team_stats = df_historical.groupby(['event_id', 'team', 'opponent', 'season'], as_index=False).agg(agg_dict).reset_index()
	#df_team_stats['cs'] = np.where(df_team_stats['gc'] <= 0, 1, 0)
	df_team_stats['pcp'] = df_team_stats['aps'] / df_team_stats['pss']
	df_opponent_stats = df_team_stats.copy()
	rename_dict = {'team':'opponent', 'opponent':'team'}
	df_opponent_stats = df_opponent_stats.rename(columns=rename_dict)
	df_opponent_stats = df_opponent_stats.drop(columns=['index', 'home'])
	feature_names = list(set(utilities.all_feature_names()) - set(['gls', 'cs']))
	rename_dict = {feature:'team_' + feature for feature in feature_names}
	df_team_stats = df_team_stats.rename(columns=rename_dict)
	rename_dict = {feature:'opp_' + feature for feature in feature_names}
	df_opponent_stats = df_opponent_stats.rename(columns=rename_dict)
	df_team_stats = pd.merge(df_team_stats, df_opponent_stats, how='inner', on=['event_id', 'team', 'opponent', 'season', 'round'])
	rename_dict = {'team_gc':'opp_gls', 'opp_gc':'team_gls'}
	df_team_stats = df_team_stats.rename(columns=rename_dict)
	df_team_stats = df_team_stats.drop(columns=['index'])
	df_team_stats = pd.concat([df_team_stats, df_opponent], axis=0, sort=True)
	return df_team_stats

def add_dgw_column(df):
	df['dgw'] = df.duplicated(subset=['player_id', 'round', 'season'], keep=False).astype(int)
	return df

def fix_player_ids(df):
	df_matching_ids = pd.read_csv(utilities.loc_name_matching, sep=',', encoding='ISO-8859-1')
	#edit player_ids for 2017 season matching with 2018 ids
	df = pd.merge(df, df_matching_ids, how='left', left_on='player_id', right_on='2017_player_id')
	df['alt_player_id'] = df['player_id'].copy()
	df['player_id'] = df['new_player_id'].fillna(df['player_id'])
	df = df.drop(['new_player_id'], axis=1)
	df = df.drop([x for x in df if x.startswith('2017_') or x.startswith('2018_') or x.endswith('_x') or x.endswith('_y')], axis=1) #get rid of extra columns
	return df

def add_monotic_round_column(df):
	counter = 1
	seasons = sorted(df['season'].unique())
	for season in seasons:
		rounds = sorted(df.loc[df['season'] == season]['round'].unique())
		for round in rounds:
			df.loc[(df['season'] == season) & (df['round'] == round), 'unique_round'] = counter
			counter+=1
	return df

def add_split_dataset_column(df):
	df_current = df.loc[df['adjusted_points'].isnull()].copy()
	df_historical = df.loc[df['adjusted_points'].notnull()].copy()

	max_unique_round = int(df_historical['unique_round'].max())
	min_unique_round = max_unique_round - NUM_TESTING_ROUNDS
	df_temp = df_historical.loc[df_historical['unique_round'] < min_unique_round].copy()
	#randomly split train and validation
	df_train, df_valid = model_selection.train_test_split(df_temp, test_size=0.2, random_state=42)	#maybe stratify on a 7+ column?
	#last 5 rounds historical are testing
	df_test = df_historical.loc[df_historical['unique_round'] >= min_unique_round].copy()
	df_current['dataset'] = 'live'
	df_train['dataset'] = 'training'
	df_valid['dataset'] = 'validation'
	for round in range(min_unique_round, max_unique_round + 1):
		df_test.loc[(df['unique_round'] == round), 'dataset'] = 'testing_' + str(round + 1 - min_unique_round)
	#df_test['dataset'] = 'testing'
	return pd.concat([df_train, df_test, df_valid, df_current], axis=0)

def create_player_stats_df():
	#first go through raw json files and append data to lists
	historical_seventeen_rows_list = []
	current_seventeen_rows_list = []
	opponent_seventeen_rows_list = []
	player_nineteen_rows_list = []

	short_name_dict = {1:'CHI', 2:'COL', 3:'CLB', 4:'DC', 5:'DAL', 6:'HOU', 7:'MTL', 8:'LA', 9:'NE', 10:'NYC', 11:'NY', 12:'ORL', 13:'PHI', 14:'POR', 15:'RSL', 16:'SJ', 17:'SEA',
	18:'SKC', 19:'TOR', 20:'VAN', 21:'ATL', 22:'MIN'}
	team_seventeen_home_dict = {}

	fixture_data = json.load(open(os.path.join(utilities.path_to_raw, utilities.subdir_seventeen, 'Fixtures.json'), 'r'))
	for entry in fixture_data:
		event_id = entry['id']
		round_id = entry['event']
		h_index = entry['team_h']
		a_index = entry['team_a']
		home_team = short_name_dict[h_index]
		away_team = short_name_dict[a_index]
		team_seventeen_home_dict[str(event_id) + ',1'] = home_team
		team_seventeen_home_dict[str(event_id) + ',0'] = away_team

	team_seventeen_home_dict['364,1'] = 'TOR'
	team_seventeen_home_dict['364,0'] = 'ATL'
	team_seventeen_home_dict['365,1'] = 'DC'
	team_seventeen_home_dict['365,0'] = 'NY'
	team_seventeen_home_dict['366,1'] = 'DAL'
	team_seventeen_home_dict['366,0'] = 'LA'
	team_seventeen_home_dict['367,1'] = 'HOU'
	team_seventeen_home_dict['367,0'] = 'CHI'
	team_seventeen_home_dict['368,1'] = 'MTL'
	team_seventeen_home_dict['368,0'] = 'NE'
	team_seventeen_home_dict['369,1'] = 'NYC'
	team_seventeen_home_dict['369,0'] = 'CLB'
	team_seventeen_home_dict['370,1'] = 'PHI'
	team_seventeen_home_dict['370,0'] = 'ORL'
	team_seventeen_home_dict['371,1'] = 'POR'
	team_seventeen_home_dict['371,0'] = 'VAN'
	team_seventeen_home_dict['372,1'] = 'RSL'
	team_seventeen_home_dict['372,0'] = 'SKC'
	team_seventeen_home_dict['373,1'] = 'SJ'
	team_seventeen_home_dict['373,0'] = 'MIN'
	team_seventeen_home_dict['374,1'] = 'SEA'
	team_seventeen_home_dict['374,0'] = 'COL'

	key_data = json.load(open(os.path.join(utilities.path_to_raw, utilities.subdir_seventeen, 'Key.json'), 'r'))
	for entry in key_data['elements']:
		player_id = entry['id']
		player_name = unidecode.unidecode((entry['first_name'] + ' ' + entry['second_name']).strip()) #remove weird characters and extra space if player has one name
		position_id = entry['element_type']

		player_data = json.load(open(os.path.join(utilities.path_to_raw, utilities.subdir_seventeen, 'Player' + str(player_id) + '.json'), 'r'))
		for player_entry in player_data['history']:
			#points = player_entry['total_points']
			mins = player_entry['minutes']
			if mins > 0:
				gls = player_entry['goals_scored']
				ass = player_entry['assists']
				cs = player_entry['clean_sheets']
				sv = player_entry['saves']
				pe = player_entry['penalties_earned']
				ps = player_entry['penalties_saved']
				pm = player_entry['penalties_missed']
				gc = player_entry['goals_conceded']
				yc = player_entry['yellow_cards']
				rc = player_entry['red_cards']
				og = player_entry['own_goals']
				oga = player_entry['own_goal_earned']
				sh = player_entry['shots']
				wf = player_entry['was_fouled']
				pss = player_entry['attempted_passes']
				aps = player_entry['completed_passes']
				pcp = aps / pss if pss > 0 else 0
				crs = player_entry['crosses']
				kp = player_entry['key_passes']
				bc = player_entry['big_chances_created']
				cl = player_entry['clearances']
				blk = player_entry['blocks']
				intc = player_entry['interceptions']
				tck = player_entry['tackles']
				br = player_entry['recoveries']
				elg = player_entry['errors_leading_to_goal']
				cost = player_entry['value'] / 10 #divide by 10 since costs are multiplied by 10 in raw json files to store it as an int instead of double

				round_id = player_entry['round']
				event_id = player_entry['fixture']
				home = int(player_entry['was_home'])
				opponent = short_name_dict[player_entry['opponent_team']]
				team = team_seventeen_home_dict[str(event_id) + ',' + str(home)]

				stats_dict = {'position_id': position_id, 'mins': mins, 'gls':gls, 'ass':ass, 'cs':cs, 'sv':sv, 'pe':pe, 'ps':ps, 'pm':pm, 'gc':gc, 'yc':yc, 'rc':rc, 'og':og, 'oga':oga, 'sh':sh, 'wf':wf, 'pss':pss, 'aps':aps, 'pcp':pcp, 'crs':crs, 'kp':kp, 'bc':bc, 'cl':cl, 'blk':blk, 'intc':intc, 'tck':tck, 'br':br, 'elg':elg}
				stats_dict = utilities.fantasy_score(stats_dict)

				player_dict = {'player_id': player_id, 'player_name':player_name, 'team':team}
				player_dict.update({'round':round_id, 'event_id':event_id, 'opponent':opponent, 'home':home, 'season':2017})
				player_dict.update(stats_dict)
				historical_seventeen_rows_list.append(player_dict)

	player_position_dict = {}
	player_eighteen_team_dict = {}
	player_nineteen_team_dict = {}
	team_id_dict = {}
	player_name_dict = {}

	hit_current_round = False

	team_data = json.load(open(os.path.join(utilities.path_to_raw, utilities.subdir_nineteen, 'Squads.json'), 'r'))
	for team_entry in team_data:
		team_id = team_entry['id']
		team_name = team_entry['short_name']
		team_id_dict[team_id] = team_name

	player_data = json.load(open(os.path.join(utilities.path_to_raw, utilities.subdir_eighteen, 'Players.json'), 'r'))
	for player_entry in player_data:
		player_id = player_entry['id']
		player_name = player_entry['known_name']
		if player_name is None:
			player_name = unidecode.unidecode((player_entry['first_name'] + ' ' + player_entry['last_name']).strip()) #remove weird characters and extra space if player has one name
		position_id = player_entry['positions'][0]
		current_cost = player_entry['cost'] / 1000000
		squad_id = player_entry['squad_id']
		squad_name = team_id_dict[squad_id]
		player_dict = {'player_id': player_id, 'player_name':player_name, 'position_id': position_id, 'cost':current_cost, 'team':squad_name}
		player_position_dict.update({player_id: position_id})
		player_eighteen_team_dict.update({player_id: squad_name})
		player_name_dict.update({player_id: player_name})

	player_data = json.load(open(os.path.join(utilities.path_to_raw, utilities.subdir_nineteen, 'Players.json'), 'r'))
	for player_entry in player_data:
		player_id = player_entry['id']
		player_name = player_entry['known_name']
		if player_name is None:
			player_name = unidecode.unidecode((player_entry['first_name'] + ' ' + player_entry['last_name']).strip()) #remove weird characters and extra space if player has one name
		position_id = player_entry['positions'][0]
		current_cost = player_entry['cost'] / 1000000
		squad_id = player_entry['squad_id']
		squad_name = team_id_dict[squad_id]
		player_dict = {'player_id': player_id, 'player_name':player_name, 'position_id': position_id, 'cost':current_cost, 'team':squad_name}
		player_nineteen_rows_list.append(player_dict)
		player_position_dict.update({player_id: position_id})
		player_nineteen_team_dict.update({player_id: squad_name})
		player_name_dict.update({player_id: player_name})

	round_data = json.load(open(os.path.join(utilities.path_to_raw, utilities.subdir_eighteen, 'Rounds.json'), 'r'))
	for match_round in round_data:
		round_id = match_round['id']
		if match_round['status'] == 'complete':# or match_round['status'] == 'active':
			matches = match_round['matches']
			for match in matches:
				match_id = match['id']
				home_squad_id = match['home_squad_id']
				away_squad_id = match['away_squad_id']
				home_squad_short_name = match['home_squad_short_name']
				away_squad_short_name = match['away_squad_short_name']
				#TODO: check if match id in id column dataframe
				match_data = json.load(open(os.path.join(utilities.path_to_raw, utilities.subdir_eighteen, 'Matches', 'Match' + str(match_id) +  '.json'), 'r'))
				for player_data in match_data:
					player_id = player_data['player_id']
					player_entry = player_data['stats']
					mins = player_entry['MIN']
					if mins > 0:
						team = player_eighteen_team_dict.get(player_id)
						position_id = player_position_dict.get(player_id)
						player_name = player_name_dict.get(player_id)
						gls = player_entry['GL']
						ass = player_entry['ASS']
						cs = player_entry['CS']
						sv = player_entry['SV']
						pe = player_entry['PE']
						ps = player_entry['PS']
						pm = player_entry['PM']
						gc = player_entry['GC']
						yc = player_entry['YC']
						rc = player_entry['RC']
						og = player_entry['OG']
						oga = player_entry['OGA']
						sh = player_entry['SH']
						wf = player_entry['WF']
						pss = player_entry['PSS']
						aps = player_entry['APS']
						pcp = aps / pss if pss > 0 else 0
						crs = player_entry['CRS']
						kp = player_entry['KP']
						bc = player_entry['BC']
						cl = player_entry['CL']
						blk = player_entry['BLK']
						intc = player_entry['INT']
						tck = player_entry['TCK']
						br = player_entry['BR']
						elg = player_entry['ELG']

						if team == home_squad_short_name:
							home = 1
							team_id = home_squad_id
							opponent_id = away_squad_id
							team = home_squad_short_name
							opponent = away_squad_short_name
						elif team == away_squad_short_name:
							home = 0
							team_id = away_squad_id
							opponent_id = home_squad_id
							team = away_squad_short_name
							opponent = home_squad_short_name
						else:
							home = 2
							team_id = away_squad_id
							opponent_id = home_squad_id
							team = away_squad_short_name
							opponent = home_squad_short_name

						stats_dict = {'position_id': position_id, 'mins': mins, 'gls':gls, 'ass':ass, 'cs':cs, 'sv':sv, 'pe':pe, 'ps':ps, 'pm':pm, 'gc':gc, 'yc':yc, 'rc':rc, 'og':og, 'oga':oga, 'sh':sh, 'wf':wf, 'pss':pss, 'aps':aps, 'pcp':pcp, 'crs':crs, 'kp':kp, 'bc':bc, 'cl':cl, 'blk':blk, 'intc':intc, 'tck':tck, 'br':br, 'elg':elg}
						stats_dict = utilities.fantasy_score(stats_dict)

						player_dict = {'player_id': player_id, 'player_name':player_name, 'team':team}
						player_dict.update({'round':round_id, 'event_id':match_id, 'opponent':opponent, 'home':home, 'season':2018})
						player_dict.update(stats_dict)
						historical_seventeen_rows_list.append(player_dict)

	round_data = json.load(open(os.path.join(utilities.path_to_raw, utilities.subdir_nineteen, 'Rounds.json'), 'r'))
	for match_round in round_data:
		round_id = match_round['id']
		if match_round['status'] == 'complete':# or match_round['status'] == 'active':
			matches = match_round['matches']
			for match in matches:
				match_id = match['id']
				home_squad_id = match['home_squad_id']
				away_squad_id = match['away_squad_id']
				home_squad_short_name = match['home_squad_short_name']
				away_squad_short_name = match['away_squad_short_name']
				#TODO: check if match id in id column dataframe
				match_data = json.load(open(os.path.join(utilities.path_to_raw, utilities.subdir_nineteen, 'Matches', 'Match' + str(match_id) +  '.json'), 'r'))
				for player_data in match_data:
					player_id = player_data['player_id']
					player_entry = player_data['stats']
					mins = player_entry['MIN']
					if mins > 0:
						team = player_nineteen_team_dict.get(player_id)
						position_id = player_position_dict.get(player_id)
						player_name = player_name_dict.get(player_id)
						gls = player_entry['GL']
						ass = player_entry['ASS']
						cs = player_entry['CS']
						sv = player_entry['SV']
						pe = player_entry['PE']
						ps = player_entry['PS']
						pm = player_entry['PM']
						gc = player_entry['GC']
						yc = player_entry['YC']
						rc = player_entry['RC']
						og = player_entry['OG']
						oga = player_entry['OGA']
						sh = player_entry['SH']
						wf = player_entry['WF']
						pss = player_entry['PSS']
						aps = player_entry['APS']
						pcp = aps / pss if pss > 0 else 0
						crs = player_entry['CRS']
						kp = player_entry['KP']
						bc = player_entry['BC']
						cl = player_entry['CL']
						blk = player_entry['BLK']
						intc = player_entry['INT']
						tck = player_entry['TCK']
						br = player_entry['BR']
						elg = player_entry['ELG']

						if team == home_squad_short_name:
							home = 1
							team_id = home_squad_id
							opponent_id = away_squad_id
							#team = home_squad_short_name
							opponent = away_squad_short_name
						elif team == away_squad_short_name:
							home = 0
							team_id = away_squad_id
							opponent_id = home_squad_id
							team = away_squad_short_name
							opponent = home_squad_short_name
						else:
							home = 2
							team_id = away_squad_id
							opponent_id = home_squad_id
							team = away_squad_short_name
							opponent = home_squad_short_name

						stats_dict = {'position_id': position_id, 'mins': mins, 'gls':gls, 'ass':ass, 'cs':cs, 'sv':sv, 'pe':pe, 'ps':ps, 'pm':pm, 'gc':gc, 'yc':yc, 'rc':rc, 'og':og, 'oga':oga, 'sh':sh, 'wf':wf, 'pss':pss, 'aps':aps, 'pcp':pcp, 'crs':crs, 'kp':kp, 'bc':bc, 'cl':cl, 'blk':blk, 'intc':intc, 'tck':tck, 'br':br, 'elg':elg}
						stats_dict = utilities.fantasy_score(stats_dict)

						player_dict = {'player_id': player_id, 'player_name':player_name, 'team':team}
						player_dict.update({'round':round_id, 'event_id':match_id, 'opponent':opponent, 'home':home, 'season':2019})
						player_dict.update(stats_dict)
						historical_seventeen_rows_list.append(player_dict)
		elif not hit_current_round: # and round_id == 27
			hit_current_round = True
			matches = match_round['matches']
			for match in matches:
				match_id = match['id']
				home_squad_id = match['home_squad_id']
				away_squad_id = match['away_squad_id']
				home_squad_short_name = match['home_squad_short_name']
				away_squad_short_name = match['away_squad_short_name']
				home_dict = {'round':round_id, 'event_id':match_id, 'opponent':away_squad_short_name, 'team':home_squad_short_name, 'home':1, 'season':2019}
				away_dict = {'round':round_id, 'event_id':match_id, 'opponent':home_squad_short_name, 'team':away_squad_short_name, 'home':0, 'season':2019}
				opponent_seventeen_rows_list.append(home_dict)
				opponent_seventeen_rows_list.append(away_dict)#create dataframes

	df_player = pd.DataFrame(player_nineteen_rows_list)
	df_historical = pd.DataFrame(historical_seventeen_rows_list)
	df_opponent = pd.DataFrame(opponent_seventeen_rows_list)

	#some home\away team and opponent info is messed up so do fix immidiately
	df_historical = df_historical.pipe(fix_player_home)

	#export aggregates of team statistics
	df_team_stats = create_team_stats_df(df_historical, df_opponent)
	df_team_stats.to_csv(utilities.loc_historical_team_stats, sep=',', index=False)

	#add players to current dataframe based on team_ids
	df_current = pd.merge(df_player, df_opponent, how='right', on=['team'])
	df_temp = pd.concat([df_historical, df_current], axis=0, sort=True) #join temporarily
	df_temp = df_temp.drop_duplicates(subset=['event_id', 'player_id', 'round', 'opponent', 'season'], keep='last')

	#add a few features
	df_temp = df_temp.pipe(add_dgw_column)
	df_temp = df_temp.pipe(fix_player_ids)
	df_temp = df_temp.pipe(add_monotic_round_column)
	df_temp = df_temp.pipe(add_split_dataset_column)
	df_temp = df_temp.sort_values(by=['unique_round'], ascending=True)

	#split up dataframes again
	df_current = df_temp.loc[df_temp['dataset'] == 'live'].copy()
	df_historical = df_temp.loc[df_temp['dataset'] != 'live'].copy()

	#drop unnecessary columns from both dataframes(adjusted points, points, etc for the current dataframe)
	df_historical = df_historical.dropna(axis=1, how='all')
	df_current = df_current.dropna(axis=1, how='all')
	return (df_historical, df_current)

if __name__ == '__main__':
	warnings.filterwarnings('ignore', category=DeprecationWarning)
	df_historical, df_current = create_player_stats_df()
	df_historical.to_csv(utilities.loc_historical_player_stats, sep=',', index=False)
	df_current.to_csv(utilities.loc_current_gameweek, sep=',', index=False)
