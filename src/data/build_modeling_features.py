import os
import json
import unidecode
import pandas as pd
import numpy as np
import sklearn
import category_encoders as encoders
#import pandas_profiling
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MinMaxScaler

subdir_seventeen = '2017data'
subdir_eighteen = '2018data'

loc_historical_output = '2018_input_data.csv'
loc_current_output = '2018_current_data.csv'
loc_clustering = '2018_clustering_output.csv'
loc_matching = 'player_ids.csv'
loc_stageing = 'staged_data.csv'

#first go through raw json files and append data to lists
historical_seventeen_rows_list = []
current_seventeen_rows_list = []
opponent_seventeen_rows_list = []
player_eighteen_rows_list = []

short_name_dict = {1:'CHI', 2:'COL', 3:'CLB', 4:'DC', 5:'DAL', 6:'HOU', 7:'MTL', 8:'LA', 9:'NE', 10:'NYC', 11:'NY', 12:'ORL', 13:'PHI', 14:'POR', 15:'RSL', 16:'SJ', 17:'SEA', 
18:'SKC', 19:'TOR', 20:'VAN', 21:'ATL', 22:'MIN'}

key_data = json.load(open(os.path.join('..', '..', 'data', 'raw', subdir_seventeen, 'Key.json'), 'r'))
for entry in key_data['elements']:
	player_id = entry['id']
	player_name = unidecode.unidecode((entry['first_name'] + ' ' + entry['second_name']).strip()) #remove weird characters and extra space if player has one name
	position_id = entry['element_type']

	player_data = json.load(open(os.path.join('..', '..', 'data', 'raw', subdir_seventeen, 'Player' + str(player_id) + '.json'), 'r'))
	for player_entry in player_data['history']:
		#points = player_entry['total_points']
		mins = player_entry['minutes']
		if mins >= 45:
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
			pcp = aps / pss
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
			att_bps = (sh // 4) + (crs // 3) + (kp // 3) + (bc) + (wf // 4) + (oga) + (pe * 2)
			def_bps = (cl // 4) + (blk // 2) + (intc // 4) + (tck // 4) + (br // 6) + (sv // 3)
			pas_bps = player_entry['pass_completion']
			if position_id == 1 or position_id == 2:
				def_pts = def_bps + (cs * 5) + (-1 * (gc // 2))
				att_pts = att_bps + (gls * 6) + (ass * 3)
			elif position_id == 3:
				def_pts = def_bps + (cs)
				att_pts = att_bps + (gls * 5) + (ass * 3)
			else:
				def_pts = def_bps
				att_pts = att_bps + (gls * 5) + (ass * 3)
			adj_points = att_pts + def_pts + pas_bps + (yc * -1) + int(mins >= 60) + 1
			real_points = adj_points + (elg * -1) + (rc * -3) + (og * -2) + (ps * 5) + (pm * -2)
			round_id = player_entry['round']
			event_id = player_entry['fixture']
			opponent = short_name_dict[player_entry['opponent_team']]
			home = int(player_entry['was_home'])
			player_dict = {'player id': player_id, 'player name':player_name, 'position id': position_id, 'points': real_points, 'mins': mins}
			player_dict.update({'adjusted points':adj_points, 'round':round_id, 'event id':event_id, 'opponent':opponent, 'home':home, 'att bps':att_bps, 'def bps':def_bps, 'pas bps':pas_bps, 'season':2017})
			historical_seventeen_rows_list.append(player_dict)

fixture_data = json.load(open(os.path.join('..', '..', 'data', 'raw', subdir_seventeen, 'Fixtures.json'), 'r'))
for entry in fixture_data:
	event_id = entry['id']
	round_id = entry['event']
	h_index = entry['team_h']
	a_index = entry['team_a']
	h_gf = entry['team_h_score']
	h_ga = entry['team_a_score']
	if h_gf is not None: #game is yet to be played
		h_gd = h_gf - h_ga
		a_gf  = h_ga
		a_ga = h_gf
		a_gd = -h_gd
		home_dict = {'gf':a_gf, 'ga':a_ga, 'gd':a_gd, 'round':round_id, 'event id':event_id, 'opponent':short_name_dict[a_index], 'team id':h_index, 'home':1, 'season':2017}
		away_dict = {'gf':h_gf, 'ga':h_ga, 'gd':h_gd, 'round':round_id, 'event id':event_id, 'opponent':short_name_dict[h_index], 'team id':a_index, 'home':0, 'season':2017}
	opponent_seventeen_rows_list.append(home_dict)
	opponent_seventeen_rows_list.append(away_dict)

#manually add in last week, forgot to scrape
opponent_seventeen_rows_list.append({'gf':2, 'ga':2, 'gd':0, 'round':33, 'event id':364, 'opponent':'TOR', 'team id':21, 'home':1, 'season':2017})
opponent_seventeen_rows_list.append({'gf':2, 'ga':2, 'gd':0, 'round':33, 'event id':364, 'opponent':'ATL', 'team id':19, 'home':0, 'season':2017})
opponent_seventeen_rows_list.append({'gf':1, 'ga':2, 'gd':-1, 'round':33, 'event id':365, 'opponent':'NY', 'team id':4, 'home':1, 'season':2017})
opponent_seventeen_rows_list.append({'gf':2, 'ga':1, 'gd':1, 'round':33, 'event id':365, 'opponent':'DC', 'team id':11, 'home':0, 'season':2017})
opponent_seventeen_rows_list.append({'gf':5, 'ga':1, 'gd':4, 'round':33, 'event id':366, 'opponent':'LA', 'team id':5, 'home':1, 'season':2017})
opponent_seventeen_rows_list.append({'gf':1, 'ga':5, 'gd':-4, 'round':33, 'event id':366, 'opponent':'DAL', 'team id':8, 'home':0, 'season':2017})
opponent_seventeen_rows_list.append({'gf':3, 'ga':0, 'gd':3, 'round':33, 'event id':367, 'opponent':'CHI', 'team id':6, 'home':1, 'season':2017})
opponent_seventeen_rows_list.append({'gf':0, 'ga':3, 'gd':-3, 'round':33, 'event id':367, 'opponent':'HOU', 'team id':1, 'home':0, 'season':2017})
opponent_seventeen_rows_list.append({'gf':2, 'ga':3, 'gd':-1, 'round':33, 'event id':368, 'opponent':'NE', 'team id':7, 'home':1, 'season':2017})
opponent_seventeen_rows_list.append({'gf':3, 'ga':2, 'gd':1, 'round':33, 'event id':368, 'opponent':'MTL', 'team id':9, 'home':0, 'season':2017})
opponent_seventeen_rows_list.append({'gf':2, 'ga':2, 'gd':0, 'round':33, 'event id':369, 'opponent':'CLB', 'team id':10, 'home':1, 'season':2017})
opponent_seventeen_rows_list.append({'gf':2, 'ga':2, 'gd':0, 'round':33, 'event id':369, 'opponent':'NYC', 'team id':3, 'home':0, 'season':2017})
opponent_seventeen_rows_list.append({'gf':6, 'ga':1, 'gd':5, 'round':33, 'event id':370, 'opponent':'ORL', 'team id':13, 'home':1, 'season':2017})
opponent_seventeen_rows_list.append({'gf':1, 'ga':6, 'gd':-5, 'round':33, 'event id':370, 'opponent':'PHI', 'team id':12, 'home':0, 'season':2017})
opponent_seventeen_rows_list.append({'gf':2, 'ga':1, 'gd':1, 'round':33, 'event id':371, 'opponent':'VAN', 'team id':14, 'home':1, 'season':2017})
opponent_seventeen_rows_list.append({'gf':1, 'ga':2, 'gd':-1, 'round':33, 'event id':371, 'opponent':'POR', 'team id':20, 'home':0, 'season':2017})
opponent_seventeen_rows_list.append({'gf':2, 'ga':1, 'gd':1, 'round':33, 'event id':372, 'opponent':'SKC', 'team id':15, 'home':1, 'season':2017})
opponent_seventeen_rows_list.append({'gf':1, 'ga':2, 'gd':-1, 'round':33, 'event id':372, 'opponent':'RSL', 'team id':18, 'home':0, 'season':2017})
opponent_seventeen_rows_list.append({'gf':3, 'ga':2, 'gd':1, 'round':33, 'event id':373, 'opponent':'MIN', 'team id':16, 'home':1, 'season':2017})
opponent_seventeen_rows_list.append({'gf':2, 'ga':3, 'gd':-1, 'round':33, 'event id':373, 'opponent':'SJ', 'team id':22, 'home':0, 'season':2017})
opponent_seventeen_rows_list.append({'gf':3, 'ga':0, 'gd':3, 'round':33, 'event id':374, 'opponent':'COL', 'team id':17, 'home':1, 'season':2017})
opponent_seventeen_rows_list.append({'gf':0, 'ga':3, 'gd':-3, 'round':33, 'event id':374, 'opponent':'SEA', 'team id':2, 'home':0, 'season':2017})

player_position_dict = {}
player_team_dict = {}
player_name_dict = {}

hit_current_round = False

player_data = json.load(open(os.path.join('..', '..', 'data', 'raw', subdir_eighteen, 'Players.json'), 'r'))
for player_entry in player_data:
	player_id = player_entry['id']
	player_name = unidecode.unidecode((player_entry['first_name'] + ' ' + player_entry['last_name']).strip()) #remove weird characters and extra space if player has one name
	position_id = player_entry['positions'][0]
	current_cost = player_entry['cost'] / 1000000
	squad_id = player_entry['squad_id']
	player_dict = {'player id': player_id, 'player name':player_name, 'position id': position_id, 'cost':current_cost, 'team id':squad_id}
	player_eighteen_rows_list.append(player_dict)
	player_position_dict.update({player_id: position_id})
	player_team_dict.update({player_id: squad_id})
	player_name_dict.update({player_id: player_name})

round_data = json.load(open(os.path.join('..', '..', 'data', 'raw', subdir_eighteen, 'Rounds.json'), 'r'))
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
			h_gf = match['home_score']
			h_ga = match['away_score']
			#print(match)
			h_gd = h_gf - h_ga
			a_gf  = h_ga
			a_ga = h_gf
			a_gd = -h_gd
			home_dict = {'gf':h_ga, 'ga':h_gf, 'gd':-h_gd, 'round':round_id, 'event id':match_id, 'opponent':away_squad_short_name, 'team id':home_squad_id, 'home':1, 'season':2018}
			away_dict = {'gf':a_ga, 'ga':a_gf, 'gd':-a_gd, 'round':round_id, 'event id':match_id, 'opponent':home_squad_short_name, 'team id':away_squad_id, 'home':0, 'season':2018}
			opponent_seventeen_rows_list.append(home_dict)
			opponent_seventeen_rows_list.append(away_dict)
			#TODO: check if match id in id column dataframe
			match_data = json.load(open(os.path.join('..', '..', 'data', 'raw', subdir_eighteen, 'Matches', 'Match' + str(match_id) +  '.json'), 'r'))
			for player_data in match_data:
				player_id = player_data['player_id']
				player_entry = player_data['stats']
				mins = player_entry['MIN']
				if mins >= 45:
					team_id = player_team_dict.get(player_id)
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
					pcp = aps / pss
					crs = player_entry['CRS']
					kp = player_entry['KP']
					bc = player_entry['BC']
					cl = player_entry['CL']
					blk = player_entry['BLK']
					intc = player_entry['INT']
					tck = player_entry['TCK']
					br = player_entry['BR']
					elg = player_entry['ELG']
					att_bps = (sh // 4) + (crs // 3) + (kp // 3) + (bc) + (wf // 4) + (oga) + (pe * 2)
					def_bps = (cl // 4) + (blk // 2) + (intc // 4) + (tck // 4) + (br // 6) + (sv // 3)
					if pcp >= .85:
						pas_bps = (pss // 35)
					else:
						pas_bps = 0
					if position_id == 1 or position_id == 2:
						def_pts = def_bps + (cs * 5) + (-1 * (gc // 2))
						att_pts = att_bps + (gls * 6) + (ass * 3)
					elif position_id == 3:
						def_pts = def_bps + (cs)
						att_pts = att_bps + (gls * 5) + (ass * 3)
					else:
						def_pts = def_bps
						att_pts = att_bps + (gls * 5) + (ass * 3)
					adj_points = att_pts + def_pts + pas_bps + (yc * -1) + int(mins >= 60) + 1
					real_points = adj_points + (elg * -1) + (rc * -3) + (og * -2) + (ps * 5) + (pm * -2)
					if team_id == home_squad_id:
						home = 1
						team_id = home_squad_id
						opponent_id = away_squad_id
						opponent = away_squad_short_name
					else:
						home = 0
						team_id = away_squad_id
						opponent_id = home_squad_id
						opponent = home_squad_short_name
					player_dict = {'player id': player_id, 'player name':player_name, 'position id': position_id, 'points': real_points, 'mins': mins, 'team id':team_id}
					player_dict.update({'adjusted points':adj_points, 'round':round_id, 'event id':match_id, 'opponent':opponent, 'home':home, 'att bps':att_bps, 'def bps':def_bps, 'pas bps':pas_bps, 'season':2018})
					historical_seventeen_rows_list.append(player_dict)
	elif not hit_current_round:
		hit_current_round = True
		matches = match_round['matches']
		for match in matches:
			match_id = match['id']
			home_squad_id = match['home_squad_id']
			away_squad_id = match['away_squad_id']
			home_squad_short_name = match['home_squad_short_name']
			away_squad_short_name = match['away_squad_short_name']
			home_dict = {'gf':-1, 'ga':-1, 'gd':-1, 'round':round_id, 'event id':match_id, 'opponent':away_squad_short_name, 'team id':home_squad_id, 'home':1, 'season':2018}
			away_dict = {'gf':-1, 'ga':-1, 'gd':-1, 'round':round_id, 'event id':match_id, 'opponent':home_squad_short_name, 'team id':away_squad_id, 'home':0, 'season':2018}
			opponent_seventeen_rows_list.append(home_dict)
			opponent_seventeen_rows_list.append(away_dict)#create dataframes

df_player = pd.DataFrame(player_eighteen_rows_list)
df_historical = pd.DataFrame(historical_seventeen_rows_list)
df_opponent = pd.DataFrame(opponent_seventeen_rows_list)
df_clustering = pd.read_csv(os.path.join('..', '..', 'data', 'processed', loc_clustering), sep=',', encoding='ISO-8859-1')
df_matching_ids = pd.read_csv(os.path.join('..', '..', 'data', 'processed', loc_matching), sep=',', encoding='ISO-8859-1')
df_staged_historical = pd.read_csv(os.path.join('..', '..', 'data', 'interim', loc_stageing), sep=',', encoding='ISO-8859-1', error_bad_lines=False)

#add players to current dataframe based on team ids
df_current = df_opponent[df_opponent['gf'] == -1]
df_current = pd.merge(df_player, df_current, how='right', on=['team id'])
df_staged_historical = df_staged_historical[df_staged_historical['cost'].isna()] #drop players yet to play in staged dataframe
df_temp = pd.concat([df_historical, df_current, df_staged_historical], axis=0, sort=True) #join temporarily
df_temp.drop_duplicates(subset=['player id', 'round', 'opponent', 'season'], keep='last', inplace=True)
df_temp.sort_values(by=['season', 'round'], ascending=True, inplace=True)
df_temp.to_csv(os.path.join('..', '..', 'data', 'interim', loc_stageing), sep=',', index=False)#often get errors when writing out

#calculate statistics about opponent
df_opponent['opp last 5 gf'] = df_opponent.groupby('opponent', as_index=False)['gf'].apply(lambda x: x.shift().rolling(window=5, min_periods=2).mean()).reset_index(level=0, drop=True)
df_opponent['opp last 5 ga'] = df_opponent.groupby('opponent', as_index=False)['ga'].apply(lambda x: x.shift().rolling(window=5, min_periods=2).mean()).reset_index(level=0, drop=True)
df_opponent['opp last 3 h adj gf'] = df_opponent.groupby(['opponent', 'home'], as_index=False)['gf'].apply(lambda x: x.shift().rolling(window=3, min_periods=2).mean()).reset_index(level=0, drop=True)
df_opponent['opp last 3 h adj ga'] = df_opponent.groupby(['opponent', 'home'], as_index=False)['ga'].apply(lambda x: x.shift().rolling(window=3, min_periods=2).mean()).reset_index(level=0, drop=True)

#binary DGW variable
df_temp['DGW'] = df_temp.duplicated(subset=['player id', 'round'], keep=False).astype(int)

#join with opponent statistics and clustering dataframes
df_opponent = df_opponent.reset_index()
df_temp = pd.merge(df_temp, df_opponent[['opp last 5 gf', 'opp last 5 ga', 'opp last 3 h adj gf', 'opp last 3 h adj ga', 'team id', 'event id', 'home']], how='left', on=['event id', 'home'])
df_temp = pd.merge(df_temp, df_clustering[['player id', 'player name', 'cluster']], how='left', on=['player id', 'player name']) #'ICA1', 'ICA2', 'tSNE1', 'tSNE2',

#edit player ids for 2017 season matching with 2018 ids
df_temp = pd.merge(df_temp, df_matching_ids, how='left', left_on='player id', right_on='2017 player id')
df_temp['alt player id'] = df_temp['player id'].copy()
df_temp['player id'] = df_temp['new player id'].fillna(df_temp['player id'])
df_temp = df_temp.drop(['new player id', 'ga', 'gd', 'gf'], axis=1)
df_temp['team id'] = df_temp['team id_y']
df_temp.drop([x for x in df_temp if x.startswith('2017 ') or x.startswith('2018 ') or x.endswith('_x') or x.endswith('_y')], axis=1, inplace=True) #get rid of extra columns

#rolling averages
df_temp.sort_values(by=['season', 'round'], ascending=True, inplace=True)
df_temp = df_temp.reset_index()
df_temp['last 3 adjusted points avg'] = df_temp.groupby(['player id'], as_index=False)['adjusted points'].apply(lambda x: x.shift().rolling(window=3, min_periods=2).mean()).reset_index(level=0, drop=True)
df_temp['last 5 adjusted points avg'] = df_temp.groupby(['player id'], as_index=False)['adjusted points'].apply(lambda x: x.shift().rolling(window=5, min_periods=2).mean()).reset_index(level=0, drop=True)
df_temp['last 5 adjusted points max'] = df_temp.groupby(['player id'], as_index=False)['adjusted points'].apply(lambda x: x.shift().rolling(window=5, min_periods=2).max()).reset_index(level=0, drop=True)
df_temp['last 5 adjusted points min'] = df_temp.groupby(['player id'], as_index=False)['adjusted points'].apply(lambda x: x.shift().rolling(window=5, min_periods=2).min()).reset_index(level=0, drop=True)
df_temp['last 5 adjusted points std dev'] = df_temp.groupby(['player id'], as_index=False)['adjusted points'].apply(lambda x: x.shift().rolling(window=5, min_periods=2).std()).reset_index(level=0, drop=True)
df_temp['last 3 h adj adjusted points avg'] = df_temp.groupby(['player id', 'home'], as_index=False)['adjusted points'].apply(lambda x: x.shift().rolling(window=3, min_periods=2).mean()).reset_index(level=0, drop=True)
df_temp['last 5 att bps avg'] = df_temp.groupby(['player id'], as_index=False)['att bps'].apply(lambda x: x.shift().rolling(window=5, min_periods=2).mean()).reset_index(level=0, drop=True)
df_temp['last 5 def bps avg'] = df_temp.groupby(['player id'], as_index=False)['def bps'].apply(lambda x: x.shift().rolling(window=5, min_periods=2).mean()).reset_index(level=0, drop=True)
df_temp['last 5 pas bps avg'] = df_temp.groupby(['player id'], as_index=False)['pas bps'].apply(lambda x: x.shift().rolling(window=5, min_periods=2).mean()).reset_index(level=0, drop=True)
#df_temp['season adjusted points avg'] = df_temp.groupby(['player id', 'season'], as_index=False)['adjusted points'].apply(lambda x: x.shift().expanding(min_periods=3).mean()).reset_index(level=0, drop=True)
#df_temp['adjusted points weighted avg'] = df_temp.groupby(['player id'], as_index=False)['adjusted points'].apply(lambda x: x.shift().ewm(span=8).mean()).reset_index(level=0, drop=True)
#df_temp['season adjusted points std dev'] = df_temp.groupby(['player id', 'season'], as_index=False)['adjusted points'].apply(lambda x: x.shift().expanding(min_periods=3).std()).reset_index(level=0, drop=True)
#df_temp['opp last 5 adjusted points conceded by cluster'] = df_temp.groupby(['cluster', 'opponent'], as_index=False)['adjusted points'].apply(lambda x: x.shift().rolling(window=5, min_periods=1).mean()).reset_index(level=0, drop=True)
df_temp['opp last 8 adjusted points conceded by cluster'] = df_temp.groupby(['cluster', 'opponent'], as_index=False)['adjusted points'].apply(lambda x: x.shift().dropna().rolling(window=8, min_periods=1).mean()).reset_index(level=0, drop=True)
df_temp['opp last 5 adjusted points conceded by position'] = df_temp.groupby(['position id', 'opponent'], as_index=False)['adjusted points'].apply(lambda x: x.shift().dropna().rolling(window=5, min_periods=1).mean()).reset_index(level=0, drop=True)
df_temp['opp last 8 adjusted points conceded by position'] = df_temp.groupby(['position id', 'opponent'], as_index=False)['adjusted points'].apply(lambda x: x.shift().dropna().rolling(window=8, min_periods=1).mean()).reset_index(level=0, drop=True)

#fill in nas
df_temp['last 3 adjusted points avg'] = df_temp.groupby(['player id'])['last 3 adjusted points avg'].fillna(method='ffill')
df_temp['last 5 adjusted points avg'] = df_temp.groupby(['player id'])['last 5 adjusted points avg'].fillna(method='ffill')
df_temp['last 5 adjusted points max'] = df_temp.groupby(['player id'])['last 5 adjusted points max'].fillna(method='ffill')
df_temp['last 5 adjusted points min'] = df_temp.groupby(['player id'])['last 5 adjusted points min'].fillna(method='ffill')
df_temp['last 5 adjusted points std dev'] = df_temp.groupby(['player id'])['last 5 adjusted points std dev'].fillna(method='ffill')
df_temp['last 3 h adj adjusted points avg'] = df_temp.groupby(['player id', 'home'])['last 3 h adj adjusted points avg'].fillna(method='ffill')
df_temp['last 5 att bps avg'] = df_temp.groupby(['player id'])['last 5 att bps avg'].fillna(method='ffill')
df_temp['last 5 def bps avg'] = df_temp.groupby(['player id'])['last 5 def bps avg'].fillna(method='ffill')
df_temp['last 5 pas bps avg'] = df_temp.groupby(['player id'])['last 5 pas bps avg'].fillna(method='ffill')
df_temp['opp last 8 adjusted points conceded by cluster'] = df_temp.groupby(['cluster', 'opponent'])['opp last 8 adjusted points conceded by cluster'].fillna(method='ffill')
df_temp['opp last 5 adjusted points conceded by position'] = df_temp.groupby(['position id', 'opponent'])['opp last 5 adjusted points conceded by position'].fillna(method='ffill')
df_temp['opp last 8 adjusted points conceded by position'] = df_temp.groupby(['position id', 'opponent'])['opp last 8 adjusted points conceded by position'].fillna(method='ffill')

#binarize position, opponent, and cluster variables
df_temp = pd.get_dummies(df_temp, columns=['position id', 'opponent', 'cluster'])

#
enc = encoders.LeaveOneOutEncoder(cols=['player id'], return_df=True)
enc.fit(df_temp, df_temp['adjusted points'])
df_temp = enc.transform(df_temp)
enc = encoders.LeaveOneOutEncoder(cols=['alt player id'], return_df=True)
enc.fit(df_temp, df_temp['adjusted points'])
df_temp = enc.transform(df_temp)
df_temp = df_temp.rename(columns={'player id': 'loo avg points season', 'alt player id': 'loo avg points'})

#split up dataframes again
df_current = df_temp.loc[(df_temp['adjusted points'].isnull())]
df_historical = df_temp.loc[(df_temp['adjusted points'].notnull())]
del df_temp

#drop unnecessary columns from both dataframes(adjusted points, points, etc for the current dataframe, cost for historical dataframe)
df_historical.dropna(axis=1, how='all', inplace=True)
df_current.dropna(axis=1, how='all', inplace=True)

#drop rows with na values from both dataframes
df_historical.dropna(axis=0, how='any', inplace=True)
df_current.dropna(axis=0, how='any', inplace=True)

#create extra boolean variables for classification
df_historical['seven plus'] = np.where(df_historical['adjusted points'] >= 7, 1, 0)
df_historical['ten plus'] = np.where(df_historical['adjusted points'] >= 10, 1, 0)

#add importance
df_historical['importance'] = MinMaxScaler().fit_transform((((df_historical['season'] - 2017) * 33 + df_historical['round'])**2).values.reshape(-1,1))

#write report out
#profile = pandas_profiling.ProfileReport(df_historical)
#profile.to_file(outputfile='datareport.html')

#write to csv
df_historical.to_csv(os.path.join('..', '..', 'data', 'processed', loc_historical_output), sep=',', index=False)
df_current.to_csv(os.path.join('..', '..', 'data', 'processed', loc_current_output), sep=',', index=False)