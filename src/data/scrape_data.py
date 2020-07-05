import datetime as dt
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import requests
import unidecode
from tqdm import tqdm

from src.utilities import config_utilities, data_utilities, logging_utilities

CURRENT_SUBDIR = data_utilities.SEASON_DIRS[data_utilities.SEASON_CURRENT]
BASE_URL = "https://fgp-data-us.s3.amazonaws.com/json/mls_mls/"

logging_utilities.setup_logging()
logger = logging.getLogger(__name__)


def setup_directories():
    dirs = [
        data_utilities.get_raw_data_filepath([CURRENT_SUBDIR, "Matches"]),
        data_utilities.get_raw_data_filepath([CURRENT_SUBDIR, "Players"]),
    ]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)


def download_data(url, headers, outfile_path, timeout=15):
    r = requests.get(url, headers=headers, timeout=timeout)
    outfile = open(outfile_path, "w")
    outfile.write(unidecode.unidecode(r.text))
    outfile.close()
    return r.status_code


def download_match_data(match_id, overwrite=False):
    url = BASE_URL + f"stats/{match_id}.json"
    headers = None
    outfile_path = data_utilities.get_raw_data_filepath(
        [CURRENT_SUBDIR, "Matches", f"Match{match_id}.json"]
    )
    if not overwrite and os.path.exists(outfile_path):
        return (200, match_id)
    return (download_data(url, headers, outfile_path), match_id)


def download_player_data(player_id):
    url = BASE_URL + f"stats/players/{player_id}.json"
    headers = None
    outfile_path = data_utilities.get_raw_data_filepath(
        [CURRENT_SUBDIR, "Players", f"Player{player_id}.json"]
    )
    return (download_data(url, headers, outfile_path), player_id)


def download_general_json():
    pages = ["Squads", "Players", "Rounds", "Venues"]
    for page in pages:
        file = f"{page}.json"
        download_data(
            BASE_URL + file.lower(),
            None,
            data_utilities.get_raw_data_filepath([CURRENT_SUBDIR, file]),
        )


def get_match_ids():
    match_ids = []
    round_data = json.load(
        open(data_utilities.get_raw_data_filepath([CURRENT_SUBDIR, "Rounds.json"]), "r")
    )
    for match_round in round_data:
        round_id = match_round["id"]
        if match_round["status"] == "complete" or match_round["status"] == "active":
            matches = match_round["matches"]
            for match in matches:
                match_id = match["id"]
                match_ids.append(match_id)
    return match_ids


def get_player_ids():
    player_data = json.load(
        open(
            data_utilities.get_raw_data_filepath([CURRENT_SUBDIR, "Players.json"]), "r"
        )
    )
    player_ids = [player_entry["id"] for player_entry in player_data]
    return player_ids


@logging_utilities.instrument_function(logger)
def download_all_data(id_type, max_connections):
    get_ids_methods = {"match": get_match_ids, "player": get_player_ids}
    download_data_methods = {
        "match": download_match_data,
        "player": download_player_data,
    }

    ids = get_ids_methods[id_type]()

    kwargs = {"total": len(ids), "unit": id_type, "unit_scale": True, "leave": True}
    with ThreadPoolExecutor(max_workers=max_connections) as executor:
        futures = (executor.submit(download_data_methods[id_type], id) for id in ids)
        for future in tqdm(as_completed(futures), **kwargs):
            try:
                status_code, id = future.result()
                tqdm.write(f"finished {id_type} id={id} status code={status_code}")
            except Exception as exc:
                tqdm.write(f"exception: {type(exc)}")


def download_all_match_data(max_connections):
    download_all_data("match", max_connections)


def download_all_player_data(max_connections):
    download_all_data("player", max_connections)


if __name__ == "__main__":
    parameters = config_utilities.get_parameter_dict(__file__)
    max_connections = parameters["max_connections"]
    setup_directories()
    download_general_json()
    download_all_match_data(max_connections)
    download_all_player_data(max_connections)
