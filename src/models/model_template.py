import os
import sys
SRC_DIRECTORY = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(SRC_DIRECTORY))
os.chdir(SRC_DIRECTORY)
from src.utilities import data_utilities as utilities
from abc import ABCMeta, abstractmethod

class ModelBase(metaclass=ABCMeta):
	MAX_MODELS = 100

	def __init__(self, season, round, target, features, model_file_suffix):
		self.season = season
		self.round = round
		self.target = target
		self.features = features
		self.model_file_suffix = model_file_suffix

	def generate_model_file_path(self):
		path_to_models = os.path.join(utilities.path_to_models, str(self.season) + '_round' + str(self.round))

		for i in range(1, self.MAX_MODELS):
			loc_model = os.path.join(path_to_models, self.model_type + '_' + str(i))
			if not os.path.exists(loc_model):
				os.makedirs(loc_model)
				self.path_to_model = loc_model
				return os.path.join(loc_model, 'model' + self.model_file_suffix)

	def get_categorical_variables(self, df):
		cat_vars = list(set(df.columns) - set(df._get_numeric_data().columns))
		return cat_vars

	@abstractmethod
	def fit(self, df_train, df_valid):
		raise NotImplementedError

	@abstractmethod
	def predict(self, df, csv_file_path):
		raise NotImplementedError

	@abstractmethod
	def predict_contrib(self, df, csv_file_path):
		raise NotImplementedError
