import logging
from typing import List

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from src.models.model_template import ModelBase
from src.models.train_cnn_ordinal_model import CNNOrdinalModel
from src.models.train_gbm_ranking_model import GBMRankingModel
from src.models.train_rnn_ordinal_model import RNNOrdinalModel
from src.models.train_robust_simple_linear_model import RobustSimpleLinearModel
from src.models.train_simple_linear_model import SimpleLinearModel
from src.utilities import data_utilities, logging_utilities

logging_utilities.setup_logging()
logger = logging.getLogger(__name__)


def run_models(model_classes: List[ModelBase]):
    for model_class in model_classes:
        model = model_class()
        model.run()


def get_all_model_predictions(model_classes: List[ModelBase]):
    df_eval = pd.DataFrame()
    df_current = pd.DataFrame()
    for model_class in model_classes:
        model = model_class()
        df_eval_tmp = pd.read_csv(
            data_utilities.get_processed_data_filepath(
                f"{model.experiment_name}_eval_predictions.csv"
            )
        )
        df_eval_tmp = df_eval_tmp.rename(
            {model.pred_column: model.experiment_name}, axis="columns"
        )
        df_current_tmp = pd.read_csv(
            data_utilities.get_processed_data_filepath(
                f"{model.experiment_name}_current_predictions.csv"
            )
        )
        df_current_tmp = df_current_tmp.rename(
            {model.pred_column: model.experiment_name}, axis="columns"
        )

        if df_eval.shape[0] == 0:
            df_eval = df_eval_tmp.copy()
            df_current = df_current_tmp.copy()
        else:
            df_eval = pd.merge(
                df_eval,
                df_eval_tmp,
                on=[
                    "player_id",
                    "player_name",
                    "opponent",
                    "round",
                    "position_id",
                    "adjusted_points",
                ],
            )
            df_current = pd.merge(
                df_current,
                df_current_tmp,
                on=[
                    "player_id",
                    "player_name",
                    "opponent",
                    "round",
                    "position_id",
                    "cost",
                ],
            )
    return (df_eval, df_current)


def get_all_experiment_names(model_classes: List[ModelBase]):
    return [model_class().experiment_name for model_class in model_classes]


def get_all_model_run_info(experiment_names: List[str]):
    df_runs = pd.DataFrame()
    for experiment_name in experiment_names:
        df_runs = df_runs.append(get_most_recent_run(experiment_name))
    df_runs = df_runs.dropna(axis=1, how="all")
    return df_runs


def get_most_recent_run(experiment_name: str):
    client = MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

    df_experiment_runs = mlflow.search_runs(experiment_ids=[experiment_id])
    df_run = df_experiment_runs.sort_values("start_time", ascending=False).head(1)
    df_run["experiment_name"] = experiment_name
    return df_run


if __name__ == "__main__":
    model_classes = [
        CNNOrdinalModel,
        GBMRankingModel,
        RNNOrdinalModel,
        RobustSimpleLinearModel,
        SimpleLinearModel,
    ]
    run_models(model_classes)
    df_eval, df_current = get_all_model_predictions(model_classes)
    df_eval.to_csv(
        data_utilities.get_processed_data_filepath("eval_predictions.csv"), index=False
    )
    df_current.to_csv(
        data_utilities.get_processed_data_filepath("current_predictions.csv"),
        index=False,
    )
    experiment_names = get_all_experiment_names(model_classes)
    df_runs = get_all_model_run_info(experiment_names)
    df_runs.to_csv(data_utilities.get_processed_data_filepath("model_runs.csv"))
