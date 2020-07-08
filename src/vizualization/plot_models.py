import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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


def plot_model_correlations(df_eval, df_current):
    plt.rcParams["font.family"] = "Gill Sans MT", "serif"
    fig, axs = plt.subplots(2, 1)
    experiment_names = [col for col in df_current.columns if "_model" in col]
    titles = ["Testing data correlation matrix", "Current round correlation matrix"]
    dfs = [df_eval, df_current]
    for i, (title, df) in enumerate(zip(titles, dfs)):
        df_corr = (
            df[experiment_names]
            .rename(
                {
                    experiment_name: experiment_name.replace("_model", "")
                    for experiment_name in experiment_names
                },
                axis=1,
            )
            .corr()
        )
        g = sns.heatmap(
            df_corr, annot=True, cmap="coolwarm", vmin=0, vmax=1, ax=axs[i],
        )
        g.set_xticklabels(g.get_xticklabels(), rotation=-15)
        axs[i].set_title(title)
        axs[i].figure.subplots_adjust(left=0.2)
    save_plot(fig, "model_correlations.png", 8, 12)


def plot_model_position_performance(df_runs):
    plt.rcParams["font.family"] = "Gill Sans MT", "serif"
    df_runs = df_runs.set_index("experiment_name")
    df_runs = df_runs[[col for col in df_runs.columns if "metrics." in col]].transpose()

    for position in ["1", "2", "3", "4", "all"]:
        metric_end = f"position.{position}" if position != "all" else ".all"
        df_runs_position = df_runs[df_runs.index.str.endswith(metric_end)].transpose()
        df_runs_position.index = df_runs_position.index.str.replace("_model", "")
        df_runs_position.columns = [
            col.split(".")[1] for col in df_runs_position.columns
        ]

        # https://stackoverflow.com/questions/54397334/annotated-heatmap-with-multiple-color-schemes
        fig, axs = plt.subplots(
            1, df_runs_position.columns.size, gridspec_kw={"wspace": 0}
        )
        for i, col in enumerate(df_runs_position.columns):
            g = sns.heatmap(
                np.array([df_runs_position[col].values]).T,
                yticklabels=df_runs_position.index,
                xticklabels=[col],
                annot=True,
                fmt=".2f",
                ax=axs[i],
                cmap="coolwarm_r",
                cbar=False,
            )
            g.set_xticklabels(g.get_xticklabels(), rotation=-15)
            if i > 0:
                axs[i].yaxis.set_ticks([])

        fig.tight_layout()
        save_plot(fig, f"position{position}_metrics.png", 8, 6)
        plt.clf()


if __name__ == "__main__":
    df_eval = pd.read_csv(
        data_utilities.get_processed_data_filepath("eval_predictions.csv")
    )
    df_current = pd.read_csv(
        data_utilities.get_processed_data_filepath("current_predictions.csv")
    )
    plot_model_correlations(df_eval, df_current)
    df_runs = pd.read_csv(data_utilities.get_processed_data_filepath("model_runs.csv"))
    plot_model_position_performance(df_runs)
