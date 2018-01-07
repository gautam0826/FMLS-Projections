import os
import json
import pandas as pd

subdir = 'data'

loc_projections = 'current predictions.csv'
loc_output = 'screened current predictions.csv'

#first go through raw json files and append data to lists
selection_rows_list = []

key_data = json.load(open(os.path.join(subdir, 'Key.json'), 'r'))
for entry in key_data['elements']:
	player_id = entry['id']
	player_name = entry['web_name']
	status = entry['status']

	player_data = json.load(open(os.path.join(subdir, 'Player' + str(player_id) + '.json'), 'r'))

	#find how much player had been playing recently
	recent_game = False #whether player has played 45+ minute game recently
	last_3_mins = 0 #mins in last 3 games
	for player_entry in player_data['history_summary']:
		mins = player_entry['minutes']
		last_3_mins += mins
		if mins >= 45:
			recent_game = True

	if status == 'a' and (recent_game or last_3_mins > 90): #check if player is not injured and played recently
			player_dict = {'player id': player_id, 'player name':player_name}
			selection_rows_list.append(player_dict)

#create dataframes
selections_df = pd.DataFrame(selection_rows_list) #only holds players likely to play
df = pd.read_csv(loc_projections, sep=',', encoding='ISO-8859-1')

#right-join predictions dataframe with selection dataframe to only keep the players who are likely to play
df = pd.merge(df, selections_df, how='right', on=['player id', 'player name'])

#write to csv
df.to_csv(loc_output, sep=',', index=False)
