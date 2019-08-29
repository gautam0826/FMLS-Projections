import pandas as pd
import numpy as np
from sklearn import model_selection
import warnings
import utilities

def filter(df):
	df_train_full = pd.read_csv(utilities.loc_train_nn_data, sep=',', encoding='ISO-8859-1')
	df_test = df_train_full.loc[df_train_full['dataset'] == 'testing'].copy()
	df_player = df_test.drop_duplicates(subset=['player_id'], keep='last')[['player_id']]
	df = pd.merge(df, df_player, how='right', on=['player_id'])
	return df

if __name__ == '__main__':
	warnings.filterwarnings('ignore', category=DeprecationWarning)
	warnings.filterwarnings('ignore', category=FutureWarning)
	df = pd.read_csv(utilities.loc_current_nn_predictions, sep=',', encoding='ISO-8859-1')
	df = df.pipe(filter)

	df = df.reset_index().groupby(['player_name', 'player_id', 'cost', 'team', 'position_id'], as_index=False).sum()
	df = df[['pred_adjusted_points', 'player_name', 'player_id', 'cost', 'team', 'position_id']]
	df.to_csv('final_projections_total.csv', index=False)
