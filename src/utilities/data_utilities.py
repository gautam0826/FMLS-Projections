import math
import os
from typing import Any, Dict, List

import pandas as pd
from sqlalchemy import create_engine

from src.utilities import config_utilities

SEASON_START = 2017
SEASON_CURRENT = 2020
SEASON_DIRS = {
    season: (f"{season}data") for season in range(SEASON_START, SEASON_CURRENT + 1)
}


def get_project_directory() -> str:
    """
    Returns project directory
            Returns:
                    project_dir (str): string containing the directory of the project
    """
    project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")
    return project_dir


def get_filepath(files: List[str]) -> str:
    return os.path.join(get_project_directory(), *files)


def get_processed_data_filepath(file: str) -> str:
    return get_filepath(["data", "processed", file])


def get_raw_data_filepath(files: List[str]) -> str:
    return get_filepath(["data", "raw"] + files)


def get_model_filepath(experiment: str, id: str) -> str:
    return get_filepath(["models", experiment, id])


def get_conf_file_path(file: str) -> str:
    local_file_path = get_filepath(["conf", "local", file])
    if os.path.exists(local_file_path):
        return local_file_path
    return get_filepath(["conf", "base", file])


def initialize_db():
    engine = create_engine("sqlite:///" + get_processed_data_filepath("fmls.db"))
    conn = engine.connect()
    return conn


# scoring logic
_att_bps_dict = {
    "sh": (1 / 4),
    "crs": (1 / 3),
    "kp": (1 / 3),
    "bc": 1,
    "wf": (1 / 4),
    "oga": 1,
    "pe": 2,
}
_def_bps_dict = {
    "cl": (1 / 4),
    "blk": (1 / 2),
    "intc": (1 / 4),
    "tck": (1 / 4),
    "br": (1 / 6),
    "sv": (1 / 3),
}
_1_2_extra_att_pts_dict = {"gls": 6, "ass": 3}
_1_2_extra_def_pts_dict = {"cs": 5, "gc": (-1 / 2)}
_3_extra_att_pts_dict = {"gls": 5, "ass": 3}
_3_extra_def_pts_dict = {"cs": 1}
_4_extra_att_pts_dict = {"gls": 5, "ass": 3}
_4_extra_def_pts_dict = {}
_extra_adj_pts_dict = {"yc": -1, "mins": (1 / 60)}
_extra_real_pts_dict = {"elg": -1, "rc": -3, "og": -2, "ps": 5, "pm": -2}


def passing_stat_names() -> List[str]:
    stat_names = ["pss", "aps", "pcp"]
    return stat_names


def attacking_bonus_stat_names() -> List[str]:
    return list(_att_bps_dict.keys())


def defending_bonus_stat_names() -> List[str]:
    return list(_def_bps_dict.keys())


def attacking_stat_names() -> List[str]:
    return attacking_bonus_stat_names() + list(_1_2_extra_att_pts_dict.keys())


def defending_stat_names() -> List[str]:
    return defending_bonus_stat_names() + list(_1_2_extra_def_pts_dict.keys())


def important_stat_names() -> List[str]:
    return (
        attacking_stat_names()
        + defending_stat_names()
        + passing_stat_names()
        + list(_extra_adj_pts_dict.keys())
    )


def all_stat_names() -> List[str]:
    return important_stat_names() + list(_extra_real_pts_dict.keys())


def minimum_important_feature_names() -> List[str]:
    return [
        "adjusted_points",
        "att_bps",
        "def_bps",
        "att_points",
        "def_points",
        "pas_bps",
        "total_bps",
    ]


def important_feature_names() -> List[str]:
    return important_stat_names() + minimum_important_feature_names()


def all_feature_names() -> List[str]:
    return all_stat_names() + minimum_important_feature_names()


def get_player_stats_columns() -> Dict[str, str]:
    column_names = all_feature_names()
    player_stats_columns = {
        column_name: "INT" if column_name != "pcp" else "FLOAT"
        for column_name in column_names
    }
    player_stats_columns.update({"cost": "FLOAT"})
    player_stats_columns.update(
        {
            column: "INT"
            for column in [
                "event_id",
                "home",
                "player_id",
                "points",
                "position_id",
                "round",
                "season",
                "unique_round",
            ]
        }
    )
    player_stats_columns.update(
        {column: "TEXT" for column in ["player_name", "team", "opponent"]}
    )
    return player_stats_columns


def fantasy_score(stats_dict: Dict[str, Any]) -> Dict[str, Any]:
    position_id = stats_dict["position_id"]
    if position_id == 1 or position_id == 2:
        _extra_att_pts_dict = _1_2_extra_att_pts_dict
        _extra_def_pts_dict = _1_2_extra_def_pts_dict
    elif position_id == 3:
        _extra_att_pts_dict = _3_extra_att_pts_dict
        _extra_def_pts_dict = _3_extra_def_pts_dict
    else:
        _extra_att_pts_dict = _4_extra_att_pts_dict
        _extra_def_pts_dict = _4_extra_def_pts_dict

    att_bps = sum(
        [
            math.floor(stats_dict[stat_type] * multiplier)
            for (stat_type, multiplier) in _att_bps_dict.items()
        ]
    )
    def_bps = sum(
        [
            math.floor(stats_dict[stat_type] * multiplier)
            for (stat_type, multiplier) in _def_bps_dict.items()
        ]
    )
    pas_bps = (stats_dict["pss"] // 35) if (stats_dict["pcp"] >= 0.85) else 0
    total_bps = att_bps + def_bps + pas_bps
    extra_att_pts = sum(
        [
            (stats_dict[stat_type] * multiplier)
            for (stat_type, multiplier) in _extra_att_pts_dict.items()
        ]
    )
    extra_def_pts = sum(
        [
            math.ceil(stats_dict[stat_type] * multiplier)
            for (stat_type, multiplier) in _extra_def_pts_dict.items()
        ]
    )
    extra_adj_pts = sum(
        [
            math.floor(stats_dict[stat_type] * multiplier)
            for (stat_type, multiplier) in _extra_adj_pts_dict.items()
        ]
    ) + int(stats_dict["mins"] > 0)
    extra_real_pts = sum(
        [
            (stats_dict[stat_type] * multiplier)
            for (stat_type, multiplier) in _extra_real_pts_dict.items()
        ]
    )
    att_pts = att_bps + extra_att_pts
    def_pts = def_bps + extra_def_pts
    adj_pts = att_pts + def_pts + pas_bps + extra_adj_pts
    real_pts = adj_pts + extra_real_pts
    stats_dict.update(
        {
            "adjusted_points": adj_pts,
            "points": real_pts,
            "att_points": att_pts,
            "def_points": def_pts,
            "att_bps": att_bps,
            "def_bps": def_bps,
            "pas_bps": pas_bps,
            "total_bps": total_bps,
        }
    )
    return stats_dict
