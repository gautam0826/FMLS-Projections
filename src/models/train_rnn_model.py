import logging

import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
from keras import callbacks, constraints, layers, models, optimizers
from sklearn import preprocessing

from src.models.model_template import ModelBase
from src.utilities import config_utilities, data_utilities, logging_utilities

logging_utilities.setup_logging()
logger = logging.getLogger(__name__)


class RNN_Ordinal_Model(ModelBase):
    def __init__(self, params, target, unused_cols, rerun_sql=True):
        super().__init__(params, target, unused_cols, rerun_sql)
        self.epochs = params.pop("epochs")
        self.batch_size = params.pop("batch_size")
        self.learning_rate = params.pop("learning_rate")
        self.normal_layer_size = params.pop("normal_layer_size")
        self.rnn_layer_size = params.pop("rnn_layer_size")
        self.upper = params.pop("upper")
        self.lower = params.pop("lower")
        self.experiment_name = "rnn_model"

    def load_training_data(self):
        data = super().load_training_data()
        self.main_input_features = [
            feature for feature in self.features if "_lag_" not in feature
        ]
        self.lag_features = [feature for feature in self.features if "_lag_" in feature]
        return data

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
        df.to_parquet(data_filepath, engine="fastparquet", compression=None)
        return df

    def prepare_x_input_dicts(self, df_dict):
        input_dicts = {}
        df_train = df_dict["train"]
        ss_main = preprocessing.StandardScaler()
        ss_main.fit(df_train[self.main_input_features])
        ss_lag = preprocessing.StandardScaler()
        ss_lag.fit(
            df_train[df_train[self.lag_features].isin([-1]).any(axis=1)][
                self.lag_features
            ]
        )

        for df_name, df in df_dict.items():
            main_input = ss_main.transform(df[self.main_input_features])
            df_lag_scaled = pd.DataFrame(
                data=ss_lag.transform(df[self.lag_features].replace({-1: np.nan})),
                columns=self.lag_features,
            )
            df_lag_scaled = df_lag_scaled.replace({np.nan: -100})

            x_input_dict = self.get_lagging_input_dict(df_lag_scaled)
            x_input_dict.update({"main_input": main_input})
            input_dicts[df_name] = x_input_dict
        return input_dicts

    def get_lagging_input_dict(self, df):
        lag_sets = ["_player", "_opp", "_team_total", "_opp_total"]
        lag_input_dict = {}

        for lag_set in lag_sets:
            # reshape into [samples, timesteps, features] for lstm
            lag_set_features = [
                feature for feature in self.lag_features if feature.endswith(lag_set)
            ]
            lag_stats_set = {
                lag_column.split("_lag_")[0] for lag_column in lag_set_features
            }
            lag_steps = int(len(lag_set_features) / len(lag_stats_set))

            lag_steps_df_list = []
            for lag_step in range(1, lag_steps):
                lag_steps_columns = [
                    "%s_lag_%g%s" % (lag_stat, lag_step, lag_set)
                    for lag_stat in sorted(lag_stats_set)
                ]
                lag_steps_df_list.append(df[lag_steps_columns].values)

            lag_input_dict[lag_set] = np.stack(
                lag_steps_df_list, axis=1
            )  # .to_numpy(copy=True)

        return {
            "lag%s_stats_input" % (lag_set): np_array
            for (lag_set, np_array) in lag_input_dict.items()
        }

    def prepare_y_input_dicts(self, df_dict):
        input_dicts = {}
        for df_name, df in df_dict.items():
            # y_input_dict = {str(quantile*100) + '_output':df[target] for quantile in quantiles}
            y_input_dict = {"main_output": df.pipe(self.prepare_ordinal_data)}
            input_dicts[df_name] = y_input_dict
        return input_dicts

    def prepare_ordinal_data(self, df):
        ordinal_df = pd.DataFrame()
        for i in range(self.lower, self.upper + 1):
            ordinal_df[f">={i}"] = np.where(df[target] >= i, 1, 0)
        return ordinal_df

    def build_model(self, X_train):
        # input layers
        main_input = layers.Input(
            shape=(len(self.main_input_features),), dtype="float32", name="main_input"
        )
        lag_player_stats_input = layers.Input(
            shape=X_train["lag_player_stats_input"].shape[1:],
            dtype="float32",
            name="lag_player_stats_input",
        )
        lag_opp_stats_input = layers.Input(
            shape=X_train["lag_opp_stats_input"].shape[1:],
            dtype="float32",
            name="lag_opp_stats_input",
        )
        lag_team_total_stats_input = layers.Input(
            shape=X_train["lag_team_total_stats_input"].shape[1:],
            dtype="float32",
            name="lag_team_total_stats_input",
        )
        lag_opp_total_stats_input = layers.Input(
            shape=X_train["lag_opp_total_stats_input"].shape[1:],
            dtype="float32",
            name="lag_opp_total_stats_input",
        )

        # masking layers
        lag_player_stats_masking = layers.Masking(mask_value=-100)(
            lag_player_stats_input
        )
        lag_opp_stats_masking = layers.Masking(mask_value=-100)(lag_opp_stats_input)
        lag_team_total_stats_masking = layers.Masking(mask_value=-100)(
            lag_team_total_stats_input
        )
        lag_opp_total_stats_masking = layers.Masking(mask_value=-100)(
            lag_opp_total_stats_input
        )

        # time series layers
        rnn_lag_player_stats = layers.GRU(
            self.rnn_layer_size, activation="relu", return_sequences=False
        )(lag_player_stats_masking)
        rnn_lag_opp_stats = layers.GRU(self.rnn_layer_size, activation="relu")(
            lag_opp_stats_masking
        )
        rnn_lag_team_total_stats = layers.GRU(self.rnn_layer_size, activation="relu")(
            lag_team_total_stats_masking
        )
        rnn_lag_opp_total_stats = layers.GRU(self.rnn_layer_size, activation="relu")(
            lag_opp_total_stats_masking
        )

        # main layers
        concat_all = layers.concatenate(
            [
                main_input,
                rnn_lag_player_stats,
                rnn_lag_opp_stats,
                rnn_lag_team_total_stats,
                rnn_lag_opp_total_stats,
            ]
        )
        concat_all = layers.Dense(
            self.normal_layer_size, activation="relu", bias_initializer="zeros"
        )(concat_all)
        concat_all = layers.Dropout(0.2)(concat_all)

        # output layers
        main_output = layers.Dense(
            self.upper - self.lower + 1,
            activation="sigmoid",
            kernel_constraint=constraints.unit_norm(axis=0),
            name="main_output",
        )(concat_all)

        # define model
        input_layers = [
            main_input,
            lag_player_stats_input,
            lag_opp_stats_input,
            lag_team_total_stats_input,
            lag_opp_total_stats_input,
        ]
        output_layers = [main_output]
        output_losses = {"main_output": "binary_crossentropy"}
        output_loss_weights = {"main_output": 1}
        opt = optimizers.Adam(lr=self.learning_rate)
        model = models.Model(inputs=input_layers, outputs=output_layers)
        model.compile(
            optimizer=opt, loss=output_losses, loss_weights=output_loss_weights
        )
        return model

    @logging_utilities.instrument_function(logger)
    def train_model(self, X_train, y_train, X_valid, y_valid):
        model = self.build_model(X_train)
        es = callbacks.EarlyStopping(monitor="loss", patience=3)
        lr_reducer = callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=2,
            verbose=1,
            mode="auto",
            min_delta=0.0001,
            cooldown=0,
            min_lr=0,
        )
        cb = [es, lr_reducer]

        history = model.fit(
            x=X_train,
            y=y_train,
            validation_data=(X_valid, y_valid),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=cb,
        )
        return model

    def load_model(self, run_id):
        model = mlflow.keras.load_model(
            data_utilities.get_model_filepath(self.experiment_name, str(run_id))
        )
        return model

    def save_model(self, model, run_id):
        mlflow.keras.save_model(
            model, data_utilities.get_model_filepath(self.experiment_name, str(run_id))
        )

    def predict(self, model, X):
        # quick fix to ensure rank monotonicity: sort output probabilities
        df_predictions = pd.DataFrame(
            data=np.sort(model.predict(X), axis=1),
            columns=[f"p(>={i})" for i in range(self.upper, self.lower - 1, -1)],
        )
        df_predictions = self.calculate_expected_value(df_predictions)
        return df_predictions

    def extract_exact_probabilities(self, df):
        df_exact_probs = pd.DataFrame()
        df_exact_probs[f"p({self.upper})"] = df[f"p(>={self.upper})"]
        for i in range(self.upper - 1, self.lower - 1, -1):
            df_exact_probs[f"p({i})"] = df[f"p(>={i})"] - df_exact_probs[f"p({i+1})"]
        return df_exact_probs

    def calculate_expected_value(self, df):
        sum = df[f"p(>={self.lower})"]
        for i in range(self.lower + 1, self.upper + 1):
            sum = sum + df[f"p(>={i})"]
        sum = sum + (self.lower - 1)
        df["expected_value"] = sum
        return df


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
        "team",
        "opponent",
        "advanced_position",
        "points",
    ]
    target = "adjusted_points"
    model = RNN_Ordinal_Model(parameters, target, unused_cols, rerun_sql=rerun_sql)
    (df_train, df_valid, df_test, df_new) = model.load_training_data()
    run_id = model.evaluate_model(df_train, df_test, df_valid)
    model.generate_current_predictions(df_train, df_test, df_valid, df_new, run_id)
