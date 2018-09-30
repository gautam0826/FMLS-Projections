import os
import pandas as pd

loc_matching_input = 'name_matching.csv'
loc_matching_output = 'player_ids.csv'

matching_df = pd.read_csv(os.path.join('..', '..', 'data', 'processed', loc_matching_input), sep=',', encoding='ISO-8859-1')
matching_df['new player id'] = matching_df['2018 player id']
matching_df.loc[matching_df['new player id'].isnull(), 'new player id'] = matching_df['2017 player id']

#check duplicates
#temp_df = matching_df[matching_df.duplicated(subset='2017 player id')]
#temp_df.to_csv(os.path.join('..', '..', 'data', 'interim', 'duplicates.csv'), sep=',', index=False)

matching_df.to_csv(os.path.join('..', '..', 'data', 'processed', loc_matching_output), sep=',', index=False)