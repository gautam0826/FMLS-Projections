import os
import json
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import minmax_scale

subdir = 'data'

loc_historical_output = 'input data.csv'
loc_current_output = 'current data.csv'
loc_clustering = 'clustering output.csv'

#first go through raw json files and append data to lists
historical_rows_list = []
current_rows_list = []
opponent_rows_list = []

key_data = json.load(open(os.path.join(subdir, 'Key.json'), 'r'))
for entry in key_data['elements']:
	player_id = entry['id']
	player_name = entry['web_name']
	position_id = entry['element_type']
	transfers_in_current = entry['transfers_in_event']
	transfers_out_current = entry['transfers_out_event']

	player_data = json.load(open(os.path.join(subdir, 'Player' + str(player_id) + '.json'), 'r'))
	for player_entry in player_data['history']:
		points = player_entry['total_points']
		mins = player_entry['minutes']
		if mins >= 45:
			eltg = player_entry['errors_leading_to_goal']
			rc = player_entry['red_cards']
			og = player_entry['own_goals']
			pk_save = player_entry['penalties_saved']
			pk_miss = player_entry['penalties_missed']
			sht = player_entry['shots']
			cr = player_entry['crosses']
			kp = player_entry['key_passes']
			bcc = player_entry['big_chances_created']
			clr = player_entry['clearances']
			blk = player_entry['blocks']
			inc = player_entry['interceptions']
			tck = player_entry['tackles']
			rec = player_entry['recoveries']
			pas = player_entry['pass_completion']
			cost = player_entry['value'] / 10 #divide by 10 since costs are multiplied by 10 in raw json files to store as int not double
			adj_points = points + (eltg) + (rc * 3) + (og * 2) + (pk_save * -5) + (pk_miss * 2)
			att_bps = (sht // 4) + (cr // 3) + (kp // 3) + (bcc)
			def_bps = (clr // 4) + (blk // 2) + (inc // 4) + (tck // 4) + (rec // 6)
			round_id = player_entry['round']
			event_id = player_entry['fixture']
			transfers_in = player_entry['transfers_in']
			transfers_out = player_entry['transfers_out']
			opponent = player_entry['opponent_team']
			home = int(player_entry['was_home'])
			player_dict = {'player id': player_id, 'player name':player_name, 'position id': position_id, 'points': points, 'mins': mins}
			player_dict.update({'adjusted points':adj_points, 'round':round_id, 'event id':event_id, 'transfers in':transfers_in, 'transfers out':transfers_out, 'opponent':opponent, 'home':home, 'cost':cost, 'att bps':att_bps, 'def bps':def_bps, 'pas bps':pas})
			historical_rows_list.append(player_dict)

	cost = entry['now_cost'] / 10 #divide by 10 since costs are multiplied by 10 in raw json files to store as int not double

	for player_entry in player_data['fixtures_summary']:
		round_id = player_entry['event']
		event_id = player_entry['id']
		home = int(player_entry['is_home'])
		if home == 1:
			opponent = player_entry['team_a']
			team = player_entry['team_h']
		else:
			opponent = player_entry['team_h']
			team = player_entry['team_a']
		player_dict = {'player id':player_id, 'player name':player_name, 'position id':position_id, 'transfers in':transfers_in_current, 'transfers out':transfers_out_current}
		player_dict.update({'cost':cost, 'round':round_id, 'event id':event_id, 'opponent':opponent, 'home':home})
		current_rows_list.append(player_dict)

fixture_data = json.load(open(os.path.join(subdir, 'Fixtures.json'), 'r'))
for entry in fixture_data:
	event_id = entry['id']
	round_id = entry['event']
	h_index = entry['team_h']
	a_index = entry['team_a']
	h_gf = entry['team_h_score']
	h_ga = entry['team_a_score']
	if h_gf is None: #game is yet to be played
		home_dict = {'gf':-1, 'ga':-1, 'gd':-1, 'round':round_id, 'event id':event_id, 'opponent':a_index, 'team id':h_index, 'home':1}
		away_dict = {'gf':-1, 'ga':-1, 'gd':-1, 'round':round_id, 'event id':event_id, 'opponent':h_index, 'team id':a_index, 'home':0}
		opponent_rows_list.append(home_dict)
		opponent_rows_list.append(away_dict)
	else:
		h_gd = h_gf - h_ga
		a_gf  = h_ga
		a_ga = h_gf
		a_gd = -h_gd
		home_dict = {'gf':a_gf, 'ga':a_ga, 'gd':a_gd, 'round':round_id, 'event id':event_id, 'opponent':a_index, 'team id':h_index, 'home':1}
		away_dict = {'gf':h_gf, 'ga':h_ga, 'gd':h_gd, 'round':round_id, 'event id':event_id, 'opponent':h_index, 'team id':a_index, 'home':0}
		opponent_rows_list.append(home_dict)
		opponent_rows_list.append(away_dict)

#create dataframes
historical_df = pd.DataFrame(historical_rows_list)
current_df = pd.DataFrame(current_rows_list)
opponent_df = pd.DataFrame(opponent_rows_list)
clustering_df = pd.read_csv(loc_clustering, sep=',', encoding='ISO-8859-1')

#calculate statistics about opponent
opponent_df['opp last 5 gf'] = opponent_df.groupby('opponent', as_index=False)['gf'].rolling(window=5, min_periods=2).mean().shift(1).reset_index(level=0, drop=True)
opponent_df['opp last 5 ga'] = opponent_df.groupby('opponent', as_index=False)['ga'].rolling(window=5, min_periods=2).mean().shift(1).reset_index(level=0, drop=True)
opponent_df['opp last 5 gd'] = opponent_df.groupby('opponent', as_index=False)['gd'].rolling(window=5, min_periods=2).mean().shift(1).reset_index(level=0, drop=True)
opponent_df['opp last 3 h adj gf'] = opponent_df.groupby(['opponent', 'home'], as_index=False)['gf'].rolling(window=3, min_periods=2).mean().shift(1).reset_index(level=0, drop=True)
opponent_df['opp last 3 h adj ga'] = opponent_df.groupby(['opponent', 'home'], as_index=False)['ga'].rolling(window=3, min_periods=2).mean().shift(1).reset_index(level=0, drop=True)
opponent_df['opp last 3 h adj gd'] = opponent_df.groupby(['opponent', 'home'], as_index=False)['gd'].rolling(window=3, min_periods=2).mean().shift(1).reset_index(level=0, drop=True)

#take out extra rows from current round dataframe
current_round = min(current_df['round'])
current_df = current_df[current_df['round'] == current_round]

#temporary dataframe used for rolling averages help
temp_df = pd.concat([historical_df, current_df], axis=0) #join temporarily

#rolling averages
temp_df = temp_df.reset_index()
temp_df['last 3 adjusted points avg'] = temp_df.groupby(['player id', 'player name'], as_index=False)['adjusted points'].rolling(window=3, min_periods=2).mean().shift(1).reset_index(level=0, drop=True)
temp_df['last 5 adjusted points avg'] = temp_df.groupby(['player id', 'player name'], as_index=False)['adjusted points'].rolling(window=5, min_periods=2).mean().shift(1).reset_index(level=0, drop=True)
temp_df['last 5 adjusted points max'] = temp_df.groupby(['player id', 'player name'], as_index=False)['adjusted points'].rolling(window=5, min_periods=2).max().shift(1).reset_index(level=0, drop=True)
temp_df['last 5 adjusted points min'] = temp_df.groupby(['player id', 'player name'], as_index=False)['adjusted points'].rolling(window=5, min_periods=2).min().shift(1).reset_index(level=0, drop=True)
temp_df['last 5 adjusted points std dev'] = temp_df.groupby(['player id', 'player name'], as_index=False)['adjusted points'].rolling(window=5, min_periods=2).std().shift(1).reset_index(level=0, drop=True)
temp_df['last 3 h adj adjusted points avg'] = temp_df.groupby(['player id', 'home'], as_index=False)['adjusted points'].rolling(window=3, min_periods=2).mean().shift(1).reset_index(level=0, drop=True)
temp_df['last 5 att bps avg'] = temp_df.groupby(['player id', 'player name'], as_index=False)['att bps'].rolling(window=5, min_periods=2).mean().shift(1).reset_index(level=0, drop=True)
temp_df['last 5 def bps avg'] = temp_df.groupby(['player id', 'player name'], as_index=False)['def bps'].rolling(window=5, min_periods=2).mean().shift(1).reset_index(level=0, drop=True)
temp_df['last 5 pas bps avg'] = temp_df.groupby(['player id', 'player name'], as_index=False)['pas bps'].rolling(window=5, min_periods=2).mean().shift(1).reset_index(level=0, drop=True)

#scale transfers in/out by round
temp_df['transfers in scaled'] = temp_df.groupby('round')['transfers in'].transform(lambda x: minmax_scale(x.astype(float)))
temp_df['transfers out scaled'] = temp_df.groupby('round')['transfers out'].transform(lambda x: minmax_scale(x.astype(float)))

#binary DGW variable
temp_df['DGW'] = temp_df.duplicated(subset=['player id', 'round'], keep=False).astype(int)

#join with opponent statistics and clustering dataframes
opponent_df = opponent_df.reset_index()
temp_df = pd.merge(temp_df, opponent_df[['opp last 5 gf', 'opp last 5 ga', 'opp last 5 gd', 'opp last 3 h adj gf', 'opp last 3 h adj ga', 'opp last 3 h adj gd', 'team id', 'event id', 'home']], how='left', on=['event id', 'home'])
temp_df = pd.merge(temp_df, clustering_df[['player id', 'player name', 'PCA1', 'PCA2', 'tSNE1', 'tSNE2', 'cluster']], how='left', on=['player id', 'player name'])

#binarize position, opponent, and cluster variables
temp_df = pd.get_dummies(temp_df, columns=['position id', 'opponent', 'cluster'])

#split up dataframes again
current_df = temp_df[temp_df['round'] == current_round]
historical_df = temp_df[temp_df['round'] != current_round]
del temp_df

#create extra boolean variables for classification
historical_df['seven plus'] = np.where(historical_df['adjusted points'] >= 7, 1, 0)
historical_df['eight plus'] = np.where(historical_df['adjusted points'] >= 8, 1, 0)
historical_df['nine plus'] = np.where(historical_df['adjusted points'] >= 9, 1, 0)
historical_df['ten plus'] = np.where(historical_df['adjusted points'] >= 10, 1, 0)

#create extra prediction variables(preformance compared to transfers in, last 5 etc.)
historical_df['adj points/transfers in'] = historical_df['adjusted points'] / (historical_df['transfers in scaled'] + 1) #add 1 to prevent divide by 0
historical_df['adj points/last 5'] = historical_df['adjusted points'] / (historical_df['last 5 adjusted points avg'] + 1) #add 1 to prevent divide by 0
historical_df['adj points/h adj last 3'] = historical_df['adjusted points'] / (historical_df['last 3 adjusted points avg'] + 1) #add 1 to prevent divide by 0

#drop unnecessary columns from current dataframe(adjusted points, points, etc)
current_df = current_df.dropna(axis=1, how='all')

#drop rows with na values from both dataframes
historical_df = historical_df.dropna(axis=0, how='any')
current_df = current_df.dropna(axis=0, how='any')

#write to csv
historical_df.to_csv(loc_historical_output, sep=',', index=False)
current_df.to_csv(loc_current_output, sep=',', index=False)
