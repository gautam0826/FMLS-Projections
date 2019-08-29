import os
import sys
SRC_DIRECTORY = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(SRC_DIRECTORY))
os.chdir(SRC_DIRECTORY)
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras import backend as K
from keras import regularizers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import MinMaxScaler
import shap
import utilities
import warnings
from src.utilities import data_utilities as utilities
from src.models.model_template import ModelBase


def eval(df):
	col_name
	df['Group'].corr(df['Age'])

if __name__ == '__main__':
	warnings.filterwarnings('ignore', category=DeprecationWarning)

	df_train_full = pd.read_csv(utilities.loc_train_nn_data, sep=',', encoding='ISO-8859-1')
	#df_test = pd.read_csv(utilities.loc_test_nn_data, sep=',', encoding='ISO-8859-1')
	df_current = pd.read_csv(utilities.loc_current_nn_data, sep=',', encoding='ISO-8859-1')
	df_train = df_train_full.loc[df_train_full['dataset'] == 'training'].copy()
	df_valid = df_train_full.loc[df_train_full['dataset'] == 'validation'].copy()
	df_test = df_train_full.loc[(df_train_full['dataset'] != 'validation') & (df_train_full['dataset'] != 'training')].copy()

	unused_cols = ['index', 'adjusted_points', 'alt_player_id', 'opponent', 'cost', 'event_id', 'player_id', 'player_name', 'round', 'team', 'season', 'unique_round', 'dataset', 'position_id', 'total_bps']
	features = [col for col in df_train.columns if col not in unused_cols]
	target = 'adjusted_points'

	season = 2019
	round = 19
	model_file_suffix = '.h5'
	model = ML_regression_nn(season, round, target, features, model_file_suffix)

	#df_train_fold = df_train.copy()
	#for test_block in sorted(df_test['dataset'].unique()):
	#	df_test_fold = df_test.loc[df_test['dataset'] == test_block].copy()
	#	model.fit(df_train_fold, df_valid)
	#	model.predict(df_test_fold, os.path.join(model.path_to_model, test_block + '.csv'))
	#	df_train_fold = pd.concat([df_train_fold, df_test_fold], axis=0, sort=True)

	model = ML_regression_nn(season, round, target, features, model_file_suffix)
	model.fit(df_train, df_valid)
	model.predict(df_current, utilities.loc_current_nn_predictions)
