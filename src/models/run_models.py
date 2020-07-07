import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.models.model_template import ModelBase
from src.models.train_cnn_ordinal_model import CNNOrdinalModel
from src.models.train_gbm_ranking_model import GBMRankingModel
from src.models.train_rnn_ordinal_model import RNNOrdinalModel
from src.models.train_robust_simple_linear_model import RobustSimpleLinearModel
from src.models.train_simple_linear_model import SimpleLinearModel
from src.utilities import data_utilities, logging_utilities

logging_utilities.setup_logging()
logger = logging.getLogger(__name__)


def save_plot(fig, file_name, length_inches, height_inches):
    fig.set_size_inches(length_inches, height_inches)
    fig.savefig(
        data_utilities.get_processed_data_filepath(f"{file_name}"),
        bbox_inches="tight",
        pad_inches=0,
        edgecolor="none",
        transparent=True,
        dpi=400,
    )


def plot_model_correlations(model_classes: List[ModelBase]):
    plt.rcParams["font.family"] = "Franklin Gothic Book", "serif"

    df_eval = pd.DataFrame()
    df_current = pd.DataFrame()
    experiment_names = []
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

        experiment_names.append(model.experiment_name)

    fig, axs = plt.subplots(2, 1)
    titles = ["Testing data correlation matrix", "Current round correlation matrix"]
    dfs = [df_eval, df_current]
    for i, (title, df) in enumerate(zip(titles, dfs)):
        axs[i].set_title(title, fontname="Franklin Gothic Medium")
        g = sns.heatmap(
            df[experiment_names].corr(),
            annot=True,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            ax=axs[i],
        )
        g.set_xticklabels(g.get_xticklabels(), rotation=-15)
        axs[i].figure.subplots_adjust(left=0.2)
    save_plot(fig, "model_correlations.png", 8, 12)


def run_models(model_classes: List[ModelBase]):
    for model_class in model_classes:
        model = model_class()
        (df_train, df_valid, df_test, df_new) = model.load_training_data()
        run_id = model.evaluate_model(df_train, df_test, df_valid)
        model.generate_current_predictions(df_train, df_test, df_valid, df_new, run_id)


if __name__ == "__main__":
    model_classes = [
        CNNOrdinalModel,
        GBMRankingModel,
        RNNOrdinalModel,
        RobustSimpleLinearModel,
        SimpleLinearModel,
    ]
    plot_model_correlations(model_classes)
