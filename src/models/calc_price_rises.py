import os
import json
import warnings
import unidecode
import pandas as pd
import numpy as np

subdir_seventeen = '2017data'
subdir_eighteen = '2018data'

loc_clustering_output = '2018 clustering input.csv'
loc_clustering_seventeen_input = '2017 clustering input.csv'

#first go through raw json files and append data to lists
stats_rows_list = []

player_data = json.load(open(os.path.join('..', '..', 'data', 'raw', subdir_eighteen, 'Players', 'Players.json'), 'r'))
for player_entry_1 in player_data:
	player_id = player_entry_1['id']
	position_id = player_entry_1['positions'][0]
	player_name = unidecode.unidecode((player_entry_1['first_name'] + ' ' + player_entry_1['last_name']).strip()) #remove weird characters and extra space if player has one name
	current_cost = player_entry_1['cost'] / 1000000

	try:
		individual_player_data = json.load(open(os.path.join('..', '..', 'data', 'raw', subdir_eighteen, 'Players', 'Player' + str(player_id) +  '.json'), 'r'))
	except:
		continue
	for individual_player_entry in individual_player_data:
		player_entry = individual_player_entry['stats']
		if len(player_entry) < 1:
			continue
		try:
			round_id = player_entry['round_id']
			continue
		except:
			mins = player_entry['MIN']
			if mins == 0:
				continue
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
			if pss > 0:
				pcp = aps / pss
			else:
				pcp = 0
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
				def_pts = def_bps + (cs * 5) * (int(mins >= 60)) + (-1 * (gc // 2))
				att_pts = att_bps + (gls * 6) + (ass * 3)
			elif position_id == 3:
				def_pts = def_bps + (cs)
				att_pts = att_bps + (gls * 5) + (ass * 3)
			else:
				def_pts = def_bps
				att_pts = att_bps + (gls * 5) + (ass * 3)
			adj_points = att_pts + def_pts + pas_bps + (yc * -1) + int(mins >= 60) + 1
			real_points = adj_points + (elg * -1) + (rc * -3) + (og * -2) + (ps * 5) + (pm * -2)
			stats_rows_list.append({'player':player_name, 'points':real_points, 'price':current_cost, 'position id':position_id})
#create dataframe
df = pd.DataFrame(stats_rows_list)
df['last 4 avg'] = df.groupby(['player'], as_index=False)['points'].apply(lambda x: x.shift().rolling(window=4, min_periods=1).mean()).reset_index(level=0, drop=True)
df['last 4'] = df['last 4 avg'] * 4
df['last 2 avg'] = df.groupby(['player'], as_index=False)['points'].apply(lambda x: x.shift().rolling(window=2, min_periods=1).mean()).reset_index(level=0, drop=True)
df = df.drop_duplicates(subset=['player'], keep='last')
df['(price / 1.5) * 5'] = df['price'] * 10 / 3
df['need'] = df['(price / 1.5) * 5'] - df['last 4']

#write to csv
df.to_csv(os.path.join('..', '..', 'data', 'processed', 'price_rises.csv'), sep=',', index=False)