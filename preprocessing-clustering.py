import os
import json
import warnings
import pandas as pd
import numpy as np

subdir = 'data'

loc_clustering_output = 'clustering input.csv'

#first go through raw json files and append data to lists
stats_rows_list = []

key_data = json.load(open(os.path.join(subdir, 'Key.json'), 'r'))
for entry in key_data['elements']:
	player_id = entry['id']
	player_name = entry['web_name']
	position_id = entry['element_type']

	goals = []
	assists = []
	shots = []
	key_passes = []
	bcc = []
	clearances = []
	blocks = []
	interceptions = []
	tackles = []
	recoveries = []
	passes_attempted = []
	passes_completed = []
	pass_completion_percentage = []
	yellow_cards = []
	red_cards = []


	player_data = json.load(open(os.path.join(subdir, 'Player' + str(player_id) + '.json'), 'r'))
	for player_entry in player_data['history']:
		points = player_entry['total_points']
		mins = player_entry['minutes']

		if mins >= 45:
			goals.append(player_entry['goals_scored'])
			assists.append(player_entry['assists'])
			shots.append(player_entry['shots'])
			key_passes.append(player_entry['key_passes'])
			bcc.append(player_entry['big_chances_created'])
			clearances.append(player_entry['clearances'])
			blocks.append(player_entry['interceptions'])
			interceptions.append(player_entry['blocks'])
			tackles.append(player_entry['tackles'])
			recoveries.append(player_entry['recoveries'])
			passes_attempted.append(player_entry['attempted_passes'])
			passes_completed.append(player_entry['completed_passes'])
			pass_completion_percentage.append((player_entry['attempted_passes'] / player_entry['completed_passes']))
			yellow_cards.append(player_entry['yellow_cards'])
			red_cards.append(player_entry['red_cards'])

	with warnings.catch_warnings(): #surpress mean of empty slice warning
		warnings.simplefilter("ignore", category=RuntimeWarning)
		stats_dict = {'player id':player_id, 'player name':player_name, 'position id':position_id, 'goals':np.mean(goals), 'assists':np.mean(assists), 'shots':np.mean(shots), 
			'key_passes':np.mean(key_passes), 'bcc':np.mean(bcc), 'clearances':np.mean(clearances), 'blocks':np.mean(blocks), 'interceptions':np.mean(interceptions),
			'tackles':np.mean(tackles), 'recoveries':np.mean(recoveries), 'passes attempted':np.mean(passes_attempted), 'passes completed':np.mean(passes_completed),
			'pass completion':np.mean(pass_completion_percentage), 'yc':np.mean(yellow_cards), 'rc':np.mean(red_cards)}
		stats_rows_list.append(stats_dict)

#create dataframe
stats_df = pd.DataFrame(stats_rows_list)

#write to csv
stats_df.to_csv(loc_clustering_output, sep=',', index=False)
