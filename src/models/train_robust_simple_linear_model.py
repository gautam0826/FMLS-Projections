import logging
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn import linear_model
from tqdm import tqdm

from src.models.model_template import ModelBase
from src.utilities import config_utilities, data_utilities, logging_utilities

logging_utilities.setup_logging()
logger = logging.getLogger(__name__)


class RobustSimpleLinearModel(ModelBase):
    def __init__(self):
        super().__init__()
        self.upper = self.params.pop("upper")
        self.lower = self.params.pop("lower")
        self.unused_cols = [
            "event_id",
            "player_id",
            "player_name",
            "unique_round",
            "cost",
            "dataset",
            "season",
            "round",
            "position_id",
            "team",
            "advanced_position",
            "opponent",
            "advanced_position+opponent",
            "home+opponent",
            "home+team",
        ]
        self.target = "adjusted_points"

    @logging_utilities.instrument_function(logger)
    def save_training_data_to_file(self, conn, data_filepath):
        # TODO: add weekday feature
        player_stats = pd.read_sql_query(
            "SELECT player_id, player_name, event_id, season, round, unique_round, opponent, team, position_id, cost, adjusted_points, home, AVG(adjusted_points) OVER (PARTITION BY player_id ORDER BY unique_round ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING) last_6_avg FROM player_stats WHERE mins >= 45 OR unique_round == (SELECT MAX(unique_round) FROM player_stats);",
            conn,
        )
        train_test_split = pd.read_sql_query("SELECT * FROM train_test_split;", conn)
        team_stats = pd.read_sql_query(
            "SELECT team, event_id, round_match_count, team_dgw, opp_dgw FROM team_stats",
            conn,
        )
        advanced_position = pd.read_sql_query(
            "SELECT player_id, season, advanced_position from advanced_position", conn
        )

        df = pd.merge(
            player_stats,
            train_test_split,
            how="left",
            on=["unique_round"],
            suffixes=("", "_y"),
        )
        df = pd.merge(
            df, team_stats, how="inner", on=["team", "event_id"], suffixes=("", "_y")
        )
        df = pd.merge(
            df,
            advanced_position,
            how="left",
            on=["player_id", "season"],
            suffixes=("", "_y"),
        )
        df = df.dropna(subset=["position_id"])
        df["last_6_avg"] = df["last_6_avg"].fillna(df["last_6_avg"].mean())
        df["advanced_position+opponent"] = (
            df["opponent"] + "_" + df["advanced_position"]
        )
        df["home+opponent"] = df["opponent"] + "_" + df["home"].map(str)
        df["home+team"] = df["team"] + "_" + df["home"].map(str)
        df[self.target] = df[self.target].clip(upper=self.upper, lower=self.lower)
        dummy_cols = [
            "team",
            "opponent",
            "advanced_position",
            "advanced_position+opponent",
            "home+opponent",
            "home+team",
        ]
        for col in dummy_cols:
            df[f"{col}_keep"] = df[col]
        df = pd.get_dummies(df, columns=dummy_cols, drop_first=True)
        df = df.rename(columns={f"{col}_keep": col for col in dummy_cols})
        df.to_parquet(data_filepath, engine="fastparquet", compression=None)
        return df

    def prepare_x_input_dicts(self, df_dict):
        min_important_round = df_dict["train"]["unique_round"].max() - 6
        df_train = df_dict.pop("train")
        df_train["sample_weight"] = np.sqrt(
            df_train["unique_round"] / min_important_round
        )
        df_train["sample_weight"] = df_train["sample_weight"].clip(upper=1)
        x_input_dict = {dataset: X[self.features] for dataset, X in df_dict.items()}
        x_input_dict["train"] = df_train[self.features + ["sample_weight"]]
        df_dict["train"] = df_train
        return x_input_dict

    def prepare_y_input_dicts(self, Y_dict):
        return {dataset: Y[self.target] for dataset, Y in Y_dict.items()}

    def build_model(self):
        model = linear_model.HuberRegressor(**self.params)
        return model

    def train_model(self, X_train, y_train, X_valid, y_valid):
        model = self.build_model()
        model = model.fit(
            X_train[self.features],
            y_train,
            sample_weight=X_train["sample_weight"].values,
        )
        return model

    def load_model(self, run_id):
        model_path = data_utilities.get_model_filepath(
            self.experiment_name, str(run_id)
        )
        model = mlflow.sklearn.load_model(model_path)
        return model

    def save_model(self, model, run_id):
        model_path = data_utilities.get_model_filepath(
            self.experiment_name, str(run_id)
        )
        mlflow.sklearn.save_model(
            model,
            model_path,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        df_coef = pd.DataFrame({"coefficients": model.coef_, "names": self.features})
        df_coef.to_csv(
            data_utilities.get_processed_data_filepath(
                f"{self.experiment_name}_coef.csv"
            ),
            index=False,
        )

    def predict(self, model, X):
        y_pred = model.predict(X)
        df_predictions = pd.DataFrame(data=y_pred, columns=[self.pred_column])
        return df_predictions


if __name__ == "__main__":
    model = RobustSimpleLinearModel()
    (df_train, df_valid, df_test, df_new) = model.load_training_data()
    run_id = model.evaluate_model(df_train, df_test, df_valid)
    model.generate_current_predictions(df_train, df_test, df_valid, df_new, run_id)
