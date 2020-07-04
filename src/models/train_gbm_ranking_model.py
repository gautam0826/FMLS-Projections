import logging

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd

from src.models.model_template import ModelBase
from src.utilities import config_utilities, data_utilities, logging_utilities

logging_utilities.setup_logging()
logger = logging.getLogger(__name__)


class GBMRankingModel(ModelBase):
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
            "team",
            "opponent",
            "advanced_position",
            "points",
        ]
        self.target = "adjusted_points"

    @logging_utilities.instrument_function(logger)
    def save_training_data_to_file(self, conn, data_filepath):
        player_stats = pd.read_sql_query(
            "SELECT player_id, player_name, event_id, season, round, unique_round, opponent, team, position_id, cost, adjusted_points, home FROM player_stats WHERE mins >= 45 OR unique_round == (SELECT MAX(unique_round) FROM player_stats);",
            conn,
        )
        player_lagging_stats = pd.read_sql_query(
            "SELECT * FROM player_lagging_stats;", conn
        )
        train_test_split = pd.read_sql_query("SELECT * FROM train_test_split;", conn)
        team_lagging_stats = pd.read_sql_query(
            "SELECT * FROM team_lagging_stats;", conn
        )
        opp_lagging_stats = pd.read_sql_query("SELECT * FROM opp_lagging_stats;", conn)
        position_vs_opponent_lagging_stats = pd.read_sql_query(
            "SELECT * FROM position_vs_opponent_lagging_stats;", conn
        )
        team_stats = pd.read_sql_query(
            "SELECT team, event_id, round_match_count, team_dgw, opp_dgw FROM team_stats",
            conn,
        )

        df = pd.merge(
            player_stats,
            train_test_split,
            how="left",
            on=["unique_round"],
            suffixes=("", "_y"),
        )
        df = pd.merge(
            df,
            team_lagging_stats,
            how="inner",
            on=["team", "opponent", "event_id"],
            suffixes=("", "_y"),
        )
        df = pd.merge(
            df,
            opp_lagging_stats,
            how="inner",
            on=["team", "opponent", "event_id"],
            suffixes=("", "_y"),
        )
        df = pd.merge(
            df,
            player_lagging_stats,
            how="inner",
            on=["player_id", "event_id"],
            suffixes=("", "_y"),
        )
        df = pd.merge(
            df,
            position_vs_opponent_lagging_stats,
            how="inner",
            on=["player_id", "event_id"],
            suffixes=("", "_y"),
        )
        df = pd.merge(
            df, team_stats, how="inner", on=["team", "event_id"], suffixes=("", "_y")
        )
        df = df.drop([x for x in df if x.endswith("_y")], axis=1)
        df = df.dropna(subset=["position_id"])
        df = df.fillna({col: -1 for col in df.columns if "_lag_" in col})
        df[self.target] = df[self.target].clip(upper=self.upper, lower=self.lower)
        df.to_parquet(data_filepath, engine="fastparquet", compression=None)
        return df

    def prepare_x_input_dicts(self, df_dict):
        df_dict["train"] = df_dict.pop("train").sort_values(["position_id"])
        df_dict["valid"] = df_dict.pop("valid").sort_values(["position_id"])
        x_input_dict = {dataset: X[self.features] for dataset, X in df_dict.items()}
        return x_input_dict

    def build_model(self):
        model = lgb.LGBMRanker(**self.params)
        return model

    @logging_utilities.instrument_function(logger)
    def train_model(self, X_train, y_train, X_valid, y_valid):
        model = self.build_model()
        group = [
            sum(X_train["position_id"] == position_id)
            for position_id in X_train["position_id"].unique()
        ]
        eval_group = [
            sum(X_valid["position_id"] == position_id)
            for position_id in X_valid["position_id"].unique()
        ]

        model.fit(
            X=X_train,
            y=y_train,
            group=group,
            eval_set=[(X_valid, y_valid)],
            eval_group=[eval_group],
        )
        return model

    def load_model(self, run_id):
        model = mlflow.lightgbm.load_model(
            data_utilities.get_model_filepath(self.experiment_name, str(run_id))
        )
        return model

    def save_model(self, model, run_id):
        # mlflow.lightgbm.save_model(
        #    model, data_utilities.get_model_filepath(self.experiment_name, str(run_id))
        # )
        pass

    def predict(self, model, X):
        y_pred = model.predict(X)
        df_predictions = pd.DataFrame(data=y_pred, columns=[self.pred_column])
        return df_predictions


if __name__ == "__main__":
    model = GBMRankingModel()
    (df_train, df_valid, df_test, df_new) = model.load_training_data()
    run_id = model.evaluate_model(df_train, df_test, df_valid)
    model.generate_current_predictions(df_train, df_test, df_valid, df_new, run_id)
