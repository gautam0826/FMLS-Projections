import logging

import h2o
import mlflow
import mlflow.h2o
import numpy as np
import pandas as pd
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

from src.models.model_template import ModelBase
from src.utilities import config_utilities, data_utilities, logging_utilities

logging_utilities.setup_logging()
logger = logging.getLogger(__name__)


class Simple_Linear_Model(ModelBase):
    def __init__(self, params, target, unused_cols, rerun_sql=True):
        super().__init__(params, target, unused_cols, rerun_sql)
        self.upper = params.pop("upper")
        self.lower = params.pop("lower")
        self.experiment_name = "simple_linear_model"
        h2o.init()

    @logging_utilities.instrument_function(logger)
    def save_training_data_to_file(self, conn, data_filepath):
        player_stats = pd.read_sql_query(
            'SELECT player_id, player_name, event_id, season, round, unique_round, opponent, team, position_id, cost, adjusted_points, home, SUBSTR("SunMonTueWedThuFriSatSun", 3*STRFTIME("%w", date) + 1, 3) AS weekday, MIN(JULIANDAY(date) - julianday(lag(date) OVER (PARTITION BY player_id ORDER BY date)), 60) as days_since_last_game, AVG(adjusted_points) OVER (PARTITION BY player_id ORDER BY unique_round ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING) last_6_avg FROM player_stats WHERE mins >= 45 OR unique_round == (SELECT MAX(unique_round) FROM player_stats);',
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
        df["days_since_last_game"] = df["days_since_last_game"].fillna(
            df["days_since_last_game"].mean()
        )
        df[self.target] = df[self.target].clip(upper=self.upper, lower=self.lower)
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
        # TODO: hierarchichal glms work from version 3.28.0.1 onwards
        interaction_pairs = [
            ("advanced_position", "opponent"),
            ("home", "team"),
            ("home", "opponent"),
        ]
        model = H2OGeneralizedLinearEstimator(
            family="poisson",
            weights_column="sample_weight",
            interaction_pairs=interaction_pairs,
            Lambda=0.0001,
            alpha=1,
        )
        return model

    def train_model(self, X_train, y_train, X_valid, y_valid):
        df_train = X_train.copy()
        df_train[self.target] = y_train
        df_valid = X_valid.copy()
        df_valid[self.target] = y_valid
        train_frame = h2o.H2OFrame(df_train)
        valid_frame = h2o.H2OFrame(df_valid)
        # train_frame[self.target] = train_frame[self.target].asfactor()
        # random_column_indexes = [train_frame.names.index(col) for col in ['advanced_position']]
        model = self.build_model()
        model.train(x=self.features, y=self.target, training_frame=train_frame)
        return model

    def load_model(self, run_id):
        model = mlflow.h2o.load_model(
            data_utilities.get_model_filepath(self.experiment_name, str(run_id))
        )
        return model

    def save_model(self, model, run_id):
        mlflow.h2o.save_model(
            model, data_utilities.get_model_filepath(self.experiment_name, str(run_id))
        )
        df_coef = model._model_json["output"]["coefficients_table"].as_data_frame()
        df_coef.to_csv(
            data_utilities.get_processed_data_filepath(
                f"{self.experiment_name}_coef.csv"
            ),
            index=False,
        )

    def predict(self, model, X):
        df_predictions = model.predict(h2o.H2OFrame(X)).as_data_frame()
        # sum = low * df_predictions[f'p{self.lower}']
        # for i in range(self.lower+1, self.upper+1):
        #    sum = sum + i * df_predictions[f'p{i}']
        # df_predictions['expected_value'] = sum
        df_predictions = df_predictions.rename(columns={"predict": self.pred_column})
        return df_predictions


if __name__ == "__main__":
    parameters = config_utilities.get_parameter_dict(__file__)
    rerun_sql = parameters.pop("rerun_sql")
    unused_cols = [
        "event_id",
        "player_id",
        "player_name",
        "unique_round",
        "cost",
        "dataset",
        "season",
        "round",
        "position_id",
    ]
    target = "adjusted_points"
    model = Simple_Linear_Model(parameters, target, unused_cols, rerun_sql=rerun_sql)
    (df_train, df_valid, df_test, df_new) = model.load_training_data()
    run_id = model.evaluate_model(df_train, df_test, df_valid)
    model.generate_current_predictions(df_train, df_test, df_valid, df_new, run_id)
    # h2o.cluster().shutdown()
