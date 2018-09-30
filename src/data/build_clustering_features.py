import os
import json
import warnings
import unidecode
import pandas as pd
import numpy as np

subdir_seventeen = '2017data'
subdir_eighteen = '2018data'

loc_clustering_output = '2018_clustering_input.csv'

#first go through raw json files and append data to lists
stats_rows_list = []

key_data = json.load(open(os.path.join('..', '..', 'data', 'raw', subdir_seventeen, 'Key.json'), 'r'))
for entry in key_data['elements']:
	player_id = entry['id']
	player_name = unidecode.unidecode((entry['first_name'] + ' ' + entry['second_name']).strip()) #remove weird characters and extra space if player has one name
	position_id = entry['element_type']

	goals = []
	assists = []
	shots = []
	key_passes = []
	bcc = []
	crosses = []
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

	player_data = json.load(open(os.path.join('..', '..', 'data', 'raw', subdir_seventeen, 'Player' + str(player_id) + '.json'), 'r'))
	for player_entry in player_data['history']:
		points = player_entry['total_points']
		mins = player_entry['minutes']

		if mins >= 45 and len(goals) < 30:
			goals.append(player_entry['goals_scored'])
			assists.append(player_entry['assists'])
			shots.append(player_entry['shots'])
			key_passes.append(player_entry['key_passes'])
			bcc.append(player_entry['big_chances_created'])
			crosses.append(player_entry['crosses'])
			clearances.append(player_entry['clearances'])
			blocks.append(player_entry['interceptions'])
			interceptions.append(player_entry['blocks'])
			tackles.append(player_entry['tackles'])
			recoveries.append(player_entry['recoveries'])
			passes_attempted.append(player_entry['completed_passes'])#looks like they're swapped in the data?
			passes_completed.append(player_entry['attempted_passes'])
			pass_completion_percentage.append((player_entry['completed_passes'] / player_entry['attempted_passes']))
			yellow_cards.append(player_entry['yellow_cards'])
			red_cards.append(player_entry['red_cards'])

	with warnings.catch_warnings(): #surpress mean of empty slice warning
		warnings.simplefilter("ignore", category=RuntimeWarning)
		stats_dict = {'player id':player_id, 'player name':player_name, 'position id':position_id, 'goals':np.mean(goals), 'assists':np.mean(assists), 'shots':np.mean(shots), 
			'key passes':np.mean(key_passes), 'bcc':np.mean(bcc), 'crosses':np.mean(crosses), 'clearances':np.mean(clearances), 'blocks':np.mean(blocks), 'interceptions':np.mean(interceptions),
			'tackles':np.mean(tackles), 'recoveries':np.mean(recoveries), 'passes attempted':np.mean(passes_attempted), 'passes completed':np.mean(passes_completed),
			'pass completion':np.mean(pass_completion_percentage), 'yc':np.mean(yellow_cards), 'rc':np.mean(red_cards), 'season':2017, 'games':len(goals), 'goals std':np.std(goals), 
			'assists std':np.std(assists), 'shots std':np.std(shots), 'key passes std':np.std(key_passes), 'bcc std':np.std(bcc), 'crosses std':np.std(crosses), 
			'clearances std':np.std(clearances), 'blocks std':np.std(blocks), 'interceptions std':np.std(interceptions), 'tackles std':np.std(tackles), 'recoveries std':np.std(recoveries),
			'passes attempted std':np.std(passes_attempted), 'passes completed std':np.std(passes_completed), 'pass completion std':np.std(pass_completion_percentage), 'yc std':np.std(yellow_cards), 
			'rc std':np.std(red_cards)}
		stats_rows_list.append(stats_dict)

player_data = json.load(open(os.path.join('..', '..', 'data', 'raw', subdir_eighteen, 'Players.json'), 'r'))
for player_entry in player_data:
	player_id = player_entry['id']
	player_name = unidecode.unidecode((player_entry['first_name'] + ' ' + player_entry['last_name']).strip()) #remove weird characters and extra space if player has one name
	position_id = player_entry['positions'][0]

	goals = []
	assists = []
	shots = []
	key_passes = []
	bcc = []
	crosses = []
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

	try:
		individual_player_data = json.load(open(os.path.join('..', '..', 'data', 'raw', subdir_eighteen, 'Players', 'Player' + str(player_id) +  '.json'), 'r'))
		for individual_player_entry in individual_player_data:
			individual_player_stats = individual_player_entry['stats']
			try:
				mins = individual_player_stats['MIN']
				try:
					round_id = individual_player_entry['round_id']
				except:
					if mins >= 45:
						goals.append(individual_player_stats['GL'])
						assists.append(individual_player_stats['ASS'])
						shots.append(individual_player_stats['SH'])
						key_passes.append(individual_player_stats['KP'])
						bcc.append(individual_player_stats['BC'])
						crosses.append(individual_player_stats['CRS'])
						clearances.append(individual_player_stats['CL'])
						blocks.append(individual_player_stats['INT'])
						interceptions.append(individual_player_stats['BLK'])
						tackles.append(individual_player_stats['TCK'])
						recoveries.append(individual_player_stats['BR'])
						passes_attempted.append(individual_player_stats['APS'])
						passes_completed.append(individual_player_stats['PSS'])
						pass_completion_percentage.append((individual_player_stats['APS'] / individual_player_stats['PSS']))
						yellow_cards.append(individual_player_stats['YC'])
						red_cards.append(individual_player_stats['RC'])
			except:
				pass

		with warnings.catch_warnings(): #surpress mean of empty slice warning
			warnings.simplefilter("ignore", category=RuntimeWarning)
			stats_dict = {'player id':player_id, 'player name':player_name, 'position id':position_id, 'goals':np.mean(goals), 'assists':np.mean(assists), 'shots':np.mean(shots), 
				'key passes':np.mean(key_passes), 'bcc':np.mean(bcc), 'crosses':np.mean(crosses), 'clearances':np.mean(clearances), 'blocks':np.mean(blocks), 'interceptions':np.mean(interceptions),
				'tackles':np.mean(tackles), 'recoveries':np.mean(recoveries), 'passes attempted':np.mean(passes_attempted), 'passes completed':np.mean(passes_completed),
				'pass completion':np.mean(pass_completion_percentage), 'yc':np.mean(yellow_cards), 'rc':np.mean(red_cards), 'season':2018, 'games':len(goals), 'goals std':np.std(goals), 
				'assists std':np.std(assists), 'shots std':np.std(shots), 'key passes std':np.std(key_passes), 'bcc std':np.std(bcc), 'crosses std':np.std(crosses), 
				'clearances std':np.std(clearances), 'blocks std':np.std(blocks), 'interceptions std':np.std(interceptions), 'tackles std':np.std(tackles), 'recoveries std':np.std(recoveries),
				'passes attempted std':np.std(passes_attempted), 'passes completed std':np.std(passes_completed), 'pass completion std':np.std(pass_completion_percentage), 'yc std':np.std(yellow_cards), 
				'rc std':np.std(red_cards)}
			stats_rows_list.append(stats_dict)
	except:
		pass

#create dataframe
df = pd.DataFrame(stats_rows_list)

#write to csv
df.to_csv(os.path.join('..', '..', 'data', 'processed', loc_clustering_output), sep=',', index=False)