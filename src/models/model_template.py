import os
from abc import ABCMeta, abstractmethod

import mlflow
import mlflow.sklearn
import pandas as pd
from tqdm import tqdm

from src.utilities import data_utilities, logging_utilities


class ModelBase(metaclass=ABCMeta):
    def __init__(self, params, target, unused_cols, rerun_sql):
        self.params = params
        self.target = target
        self.unused_cols = unused_cols
        self.rerun_sql = rerun_sql
        self.pred_column = "expected_value"

    def load_training_data(self):
        data_filepath = data_utilities.get_processed_data_filepath(
            f"{self.experiment_name}_data.parquet"
        )
        if self.rerun_sql:
            df = self.save_training_data_to_file(
                conn=data_utilities.initialize_db(), data_filepath=data_filepath
            )
        else:
            df = pd.read_parquet(data_filepath)
        self.features = [
            col
            for col in df.columns
            if col not in self.unused_cols and col != self.target
        ]
        df[self.target] = df[self.target].clip(upper=self.upper, lower=self.lower)
        df_train = df.loc[df["dataset"] == "training"].copy()
        df_valid = df.loc[df["dataset"] == "validation"].copy()
        df_test = df.loc[df["dataset"] == "testing"].copy()
        df_new = df.loc[df["dataset"] == "live"].copy()
        return (df_train, df_valid, df_test, df_new)

    @abstractmethod
    def save_training_data_to_file(self, conn):
        raise NotImplementedError

    def prepare_x_input_dicts(self, X_dict):
        return {dataset: X[self.features] for dataset, X in X_dict.items()}

    def prepare_y_input_dicts(self, Y_dict):
        return {dataset: Y[self.target] for dataset, Y in Y_dict.items()}

    def get_categorical_variables(self, df):
        cat_vars = list(set(df.columns) - set(df._get_numeric_data().columns))
        return cat_vars

    def get_most_recent_run(self):
        experiment_model_dir = os.path.join(
            data_utilities.path_to_models, self.experiment_name
        )
        model_dirs = [
            os.path.join(experiment_model_dir, d)
            for d in os.listdir(experiment_model_dir)
            if os.path.isdir(os.path.join(experiment_model_dir, d))
        ]
        latest_model_path = max(model_dirs, key=os.path.getmtime)
        return latest_model_path

    @abstractmethod
    def build_model(self):
        raise NotImplementedError

    @abstractmethod
    def train_model(self, X_train, y_train, X_valid, y_valid):
        raise NotImplementedError

    @abstractmethod
    def load_model(self, run_id):
        raise NotImplementedError

    @abstractmethod
    def save_model(self, model, run_id):
        raise NotImplementedError

    @abstractmethod
    def predict(self, model, X):
        raise NotImplementedError

    def calculate_metrics(self, df_test, count_players, suffix=""):
        metrics = {}
        metrics[f"kendall_tau_correlation{suffix}"] = df_test["adjusted_points"].corr(
            df_test[self.pred_column], method="kendall"
        )
        metrics[f"spearman_rho_correlation{suffix}"] = df_test["adjusted_points"].corr(
            df_test[self.pred_column], method="spearman"
        )
        # metrics[f"dcg_score{suffix}"] = dcg_score(
        #    df_test["adjusted_points"], df_test[self.pred_column]
        # )
        sum = 0
        num_hits = 0
        for round in df_test["round"].unique():
            df_round = (
                df_test.loc[df_test["round"] == round]
                .sort_values(self.pred_column, ascending=False)
                .head(count_players)
            )
            sum = sum + df_round["adjusted_points"].sum()
            num_hits = num_hits + df_round.loc[df_round["adjusted_points"] > 7].shape[0]
        num_testing_weeks = len(df_test["round"].unique().tolist())
        metrics[f"top_{count_players}_avg{suffix}"] = sum / (
            num_testing_weeks * count_players
        )
        metrics[f"top_{count_players}_hit_rate{suffix}"] = num_hits / (
            num_testing_weeks * count_players
        )
        return metrics

    def get_experiment_info(self, model, df_test):
        params = self.params
        # params.update({'features':self.features})
        metrics = self.calculate_metrics(df_test, 15, ".all")
        position_counts = {1: 3, 2: 6, 3: 6, 4: 4}
        for position_id, count_players in position_counts.items():
            df_position = df_test.loc[df_test["position_id"] == position_id]
            metrics.update(
                self.calculate_metrics(
                    df_position, count_players, f".position.{position_id}"
                )
            )
        artifacts = {}
        # tags = {
        #    "metaflow_runid" : current.run_id,
        #    "username" : current.username,
        #    "stepname" : current.step_name,
        #    "taskid" : current.task_id
        # }
        return (params, metrics, artifacts)

    def log_experiment(self, model, df_test):
        params, metrics, artifacts = self.get_experiment_info(model, df_test)
        mlflow_uri = os.path.join(
            "file:/" + data_utilities.get_project_directory(), "mlruns"
        )
        mlflow.set_tracking_uri(mlflow_uri)

        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            # mlflow.log_artifacts(artifacts)
            # mlflow.set_tags(tags)
            # mlflow.sklearn.log_model(model, "model")

        return run_id

    def evaluate_model(self, df_train, df_test, df_valid):
        df_predictions = pd.DataFrame()
        df_train_fold = df_train.copy()
        for test_block in tqdm(sorted(df_test["unique_round"].unique())):
            df_test_fold = df_test.loc[df_test["unique_round"] == test_block].copy()

            # prepare data
            df_dict = {"train": df_train_fold, "test": df_test_fold, "valid": df_valid}
            input_x_dicts = self.prepare_x_input_dicts(df_dict)
            input_y_dicts = self.prepare_y_input_dicts(df_dict)

            model = self.train_model(
                input_x_dicts["train"],
                input_y_dicts["train"],
                input_x_dicts["valid"],
                input_y_dicts["valid"],
            )

            df_predictions_fold = self.predict(model, input_x_dicts["test"])
            df_predictions_fold = pd.concat(
                [
                    df_test_fold[
                        [
                            "player_id",
                            "player_name",
                            "opponent",
                            "round",
                            "position_id",
                            "adjusted_points",
                        ]
                    ].reset_index(drop=True),
                    df_predictions_fold.reset_index(drop=True),
                ],
                axis=1,
            )
            df_predictions = pd.concat([df_predictions, df_predictions_fold], axis=0)
            df_train_fold = pd.concat([df_train_fold, df_test_fold], axis=0, sort=True)

        df_predictions.to_csv(
            data_utilities.get_processed_data_filepath(
                f"{self.experiment_name}_eval_predictions.csv"
            ),
            index=False,
        )
        run_id = self.log_experiment(model, df_predictions)
        return run_id

    def generate_current_predictions(self, df_train, df_test, df_valid, df_new, run_id):
        df_train = pd.concat([df_train, df_test], axis=0)
        df_dict = {"train": df_train, "valid": df_valid, "new": df_new}
        input_x_dicts = self.prepare_x_input_dicts(df_dict)
        df_dict.pop("new")
        input_y_dicts = self.prepare_y_input_dicts(df_dict)

        model = self.train_model(
            input_x_dicts["train"],
            input_y_dicts["train"],
            input_x_dicts["valid"],
            input_y_dicts["valid"],
        )

        df_predictions = self.predict(model, input_x_dicts["new"])
        df_predictions = pd.concat(
            [
                df_new[
                    [
                        "player_id",
                        "player_name",
                        "opponent",
                        "round",
                        "position_id",
                        "cost",
                    ]
                ].reset_index(drop=True),
                df_predictions.reset_index(drop=True),
            ],
            axis=1,
        )
        df_predictions.to_csv(
            data_utilities.get_processed_data_filepath(
                f"{self.experiment_name}_current_predictions.csv"
            ),
            index=False,
        )
        self.save_model(model, run_id)
