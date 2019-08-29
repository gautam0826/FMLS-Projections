import os
import sys
SRC_DIRECTORY = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(SRC_DIRECTORY))
os.chdir(SRC_DIRECTORY)
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, Reshape, Concatenate
from keras.layers.embeddings import Embedding
from keras import backend as K
from keras import regularizers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import MinMaxScaler
import shap
import warnings
from src.utilities import data_utilities as utilities
from src.models.model_template import ModelBase

class ML_regression_nn(ModelBase):
	model = None
	scaler = None
	UNITS = 512

	def __init__(self, season, round, target, features, model_file_suffix, quantile=0.5, scale=True):
		super().__init__(season, round, target, features, model_file_suffix)
		self.model_type = 'ML_regression'
		self.quantile = quantile
		self.file_path = self.generate_model_file_path()
		self.scaler = MinMaxScaler(feature_range = (0, 1)) if scale else None

	def create_model(self, df):
		def tilted_loss(q,y,f):
			e = (y-f)
			return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)
		#create model

		#cat_vars = self.get_categorical_variables(df)
		#cont_vars = [var for var in self.features if var not in cat_vars]

		#cat_sizes = {}
		#cat_embsizes = {}
		#for cat in cat_vars:
		#	cat_sizes[cat] = df[cat].nunique()
		#	cat_embsizes[cat] = min(50, cat_sizes[cat]//2+1)

		#concat = []
		#ins = []
		#for cat in cat_vars:
		#	x = Input((1,), name=cat)
		#	ins.append(x)
		#	x = Embedding(cat_sizes[cat]+1, cat_embsizes[cat], input_length=1)(x)
		#	x = Reshape((cat_embsizes[cat],))(x)
		#	concat.append(x)

		#y = Input((len(cont_vars),), name='cont_vars')
		#ins.append(y)
		#concat.append(y)
		#y = Concatenate()(concat)
		#y = Dense(100, activation= 'relu')(y)
		#y = Dense(1)(y)
		#model = Model(ins, y)
		#model.compile('adam', loss=lambda y,f: tilted_loss(self.quantile,y,f))

		model = Sequential([
			Dense(self.UNITS, activation='relu', input_dim=len(self.features), bias_initializer='zeros'),
			BatchNormalization(),
			Dense(self.UNITS, activation='relu'),
			Dropout(0.25),
			Dense(self.UNITS, activation='relu'),
			Dropout(0.25),
			Dense(1, kernel_initializer='normal')#, kernel_regularizer=regularizers.l2(0.01))
		])
		#compile model
		opt = Adam(lr=0.001)
		model.compile(loss=lambda y,f: tilted_loss(self.quantile,y,f), optimizer=opt)
		return model

	def fit(self, df_train, df_valid):
		self.model = self.create_model(df_train)
		es = EarlyStopping(monitor='loss', patience=3)
		lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2,
										verbose=1, mode='auto', min_delta=0.0001, cooldown=0,
										min_lr=0)
		checkpoint = ModelCheckpoint(filepath=self.file_path, monitor='val_loss',
										save_best_only=True)

		callbacks = [es, lr_reducer, checkpoint]

		X_train = df_train[self.features]
		X_valid = df_valid[self.features]
		y_train = df_train[self.target]
		y_valid = df_valid[self.target]
		if self.scaler is not None:
			self.scaler.fit(X_train)
			X_train = self.scaler.transform(X_train)
			X_valid = self.scaler.transform(X_valid)
		self.model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=64, epochs=16, callbacks=callbacks)

	def predict(self, df, csv_file_path):
		X = df[self.features]
		if self.scaler is not None:
			X = self.scaler.transform(X)
		df['pred_' + self.model_type + '_' + self.target + '(' + str(self.quantile) + ')'] = self.model.predict(X)
		df.to_csv(csv_file_path, sep=',', index=False)

	def predict_contrib(self, df, csv_file_path):
		X = df[self.features]
		explainer = shap.DeepExplainer(self.model, X)
		#feature_contributions = model.predict(X_current, pred_contrib=True)
		#feature_contributions_columns = [col + '_shap_value' for col in X.columns]
		#feature_contributions_columns.append('intercept_shap_value')
		#df_contrib = pd.DataFrame(feature_contributions, columns=feature_contributions_columns)
		df['pred_' + self.model_type + '_' + self.target + '(' + str(self.quantile) + ')'] = self.model.predict(X)
		df.to_csv(csv_file_path, sep=',', index=False)

if __name__ == '__main__':
	warnings.filterwarnings('ignore', category=DeprecationWarning)

	df_train_full = pd.read_csv(utilities.loc_train_nn_data, sep=',', encoding='ISO-8859-1')
	df_test = pd.read_csv(utilities.loc_test_nn_data, sep=',', encoding='ISO-8859-1')
	df_current = pd.read_csv(utilities.loc_current_nn_data, sep=',', encoding='ISO-8859-1')
	df_train = df_train_full.loc[df_train_full['dataset'] == 'training'].copy()
	df_valid = df_train_full.loc[df_train_full['dataset'] == 'validation'].copy()

	unused_cols = ['index', 'adjusted_points', 'alt_player_id', 'opponent', 'cost', 'event_id', 'player_id', 'player_name', 'round', 'team', 'season', 'unique_round', 'dataset', 'position_id', 'total_bps']
	features = [col for col in df_train.columns if col not in unused_cols]
	target = 'adjusted_points'

	#TODO do this programatically
	season = 2019
	round = 20
	model_file_suffix = '.h5'
	model = ML_regression_nn(season, round, target, features, model_file_suffix)

	#TODO move this to testing file
	#df_train_fold = df_train.copy()
	#for test_block in sorted(df_test['dataset'].unique()):
	#	df_test_fold = df_test.loc[df_test['dataset'] == test_block].copy()
	#	model.fit(df_train_fold, df_valid)
	#	model.predict(df_test_fold, os.path.join(model.path_to_model, test_block + '.csv'))
	#	df_train_fold = pd.concat([df_train_fold, df_test_fold], axis=0, sort=True)

	model = ML_regression_nn(season, round, target, features, model_file_suffix)
	model.fit(df_train, df_valid)
	model.predict(df_current, utilities.loc_current_nn_predictions)
