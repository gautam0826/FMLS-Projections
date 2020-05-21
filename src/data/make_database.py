import json
import logging
import os

import pandas as pd
import unidecode
from jinja2 import Template
from sqlalchemy import create_engine

from src.utilities import config_utilities, data_utilities, logging_utilities

CREATE_TEMPLATE = """
CREATE TABLE IF NOT EXISTS player_stats (
{% for column, column_type in player_stats_columns.items() %}{{column}} {{column_type}}{% if not loop.last %}, {% endif %}{% endfor %})
;
"""
INSERT_TEMPLATE = """INSERT INTO player_stats VALUES (?{% for _ in range (1, n_columns) %}, ?{% endfor %});"""
TEAM_STATS_VIEW_TEMPLATE = """
CREATE VIEW team_stats AS
WITH team_stats_cte AS (
SELECT event_id, team, opponent, unique_round, {% for column, agg in agg_dict.items() %}{{agg}} AS team_{{column}}{% if not loop.last %}, {% endif %}{% endfor %}
FROM player_stats
GROUP BY event_id, team),
opp_stats_cte AS (
SELECT event_id, opponent AS team, team AS opponent, {% for column in agg_dict.keys() %}team_{{column}} AS opp_{{column}}{% if not loop.last %}, {% endif %}{% endfor %}
FROM team_stats_cte),
round_match_count AS (
SELECT unique_round, COUNT(DISTINCT event_id) AS round_match_count from player_stats
GROUP BY unique_round
),
team_dgw AS (
SELECT unique_round, team, COUNT(*) - 1 AS team_dgw
FROM team_stats_cte
GROUP BY unique_round, team
),
opp_dgw AS (
SELECT unique_round, team AS opponent, team_dgw as opp_dgw
FROM team_dgw
)
SELECT team_stats_cte.event_id, team_stats_cte.opponent, team_stats_cte.team, team_stats_cte.unique_round, team_dgw.team_dgw, opp_dgw.opp_dgw, round_match_count.round_match_count, {% for column in agg_dict.keys() %}team_{{column}}, opp_{{column}}{% if not loop.last %}, {% endif %}{% endfor %} FROM team_stats_cte
INNER JOIN opp_stats_cte ON team_stats_cte.event_id = opp_stats_cte.event_id AND team_stats_cte.team = opp_stats_cte.team AND team_stats_cte.opponent = opp_stats_cte.opponent
INNER JOIN round_match_count ON team_stats_cte.unique_round = round_match_count.unique_round
INNER JOIN team_dgw ON team_stats_cte.unique_round = team_dgw.unique_round AND team_stats_cte.team = team_dgw.team
INNER JOIN opp_dgw ON team_stats_cte.unique_round = opp_dgw.unique_round AND team_stats_cte.opponent = opp_dgw.opponent
;
"""
NAME_MATCHING_VIEW_TEMPLATE = """
CREATE VIEW name_matching AS
WITH player_info_2017 AS (
SELECT DISTINCT player_name AS player_name_2017, player_id AS player_id_2017, season FROM player_stats WHERE season = 2017
),
player_info_2018 AS (
SELECT DISTINCT player_name AS player_name_2018, player_id AS player_id_2018, season FROM player_stats WHERE season > 2017
)
SELECT player_name_2017, player_name_2018, player_id_2017, player_id_2018, 2018 AS season
FROM player_info_2018
LEFT JOIN player_info_2017 ON player_info_2018.player_name_2018 = player_info_2017.player_name_2017 OR {% for a, b in player_id_match_dict.items() %}(player_info_2017.player_id_2017 = {{a}} AND player_info_2018.player_id_2018 = {{b}}){% if not loop.last %} OR {% endif %}{% endfor %}
;
"""
UPDATE_TRAIN_TEST_SPLIT = """
WITH rounds AS (
SELECT DISTINCT unique_round
FROM player_stats
ORDER BY RANDOM()
)
INSERT INTO train_test_split
SELECT unique_round,
CASE
    WHEN unique_round == (SELECT MAX(unique_round) FROM rounds) THEN 'live'
    WHEN unique_round > (SELECT MAX(unique_round) FROM rounds) - {{num_testing_rounds+1}} THEN 'testing'
    WHEN ROW_NUMBER() OVER () <= (SELECT MAX(unique_round) FROM rounds) * 0.2 THEN 'validation'
    ELSE 'training'
END AS dataset
FROM rounds
;
"""

logging_utilities.setup_logging()
logger = logging.getLogger(__name__)


def execute_templated_sql_script(conn, template, template_args_dict):
    sql_script = Template(template).render(**template_args_dict)
    # logger.critical(sql_script)
    conn.execute(sql_script)
    conn.connection.commit()


def create_player_stats_table(conn):
    template_args_dict = {
        "player_stats_columns": data_utilities.get_player_stats_columns()
    }
    execute_templated_sql_script(conn, CREATE_TEMPLATE, template_args_dict)


def insert_all_data(conn):
    conn.execute("DELETE FROM player_stats WHERE cost IS NOT NULL;")
    for season in range(data_utilities.SEASON_START, data_utilities.SEASON_CURRENT + 1):
        insert_season_data(conn, season)


def insert_season_data(conn, season):
    if season <= 2017:
        _insert_2017_season_data(conn, season)
    else:
        _insert_2018_season_data(conn, season)
    _insert_monotonic_round_column(conn, season)


def get_saved_rounds(conn, season):
    # query the highest round for a given season
    cur = conn.connection.cursor()
    cur.execute(f"SELECT DISTINCT round FROM player_stats WHERE season = {season}")
    return [round[0] for round in cur.fetchall()]


@logging_utilities.instrument_function(logger)
def _insert_2017_season_data(conn, season):
    season_subdir = data_utilities.SEASON_DIRS[season]
    saved_rounds = get_saved_rounds(conn, season)

    rows_list = []
    # home_dict is hard coded for the last week due to a scraping error
    short_name_dict = {
        1: "CHI",
        2: "COL",
        3: "CLB",
        4: "DC",
        5: "DAL",
        6: "HOU",
        7: "MTL",
        8: "LA",
        9: "NE",
        10: "NYC",
        11: "NY",
        12: "ORL",
        13: "PHI",
        14: "POR",
        15: "RSL",
        16: "SJ",
        17: "SEA",
        18: "SKC",
        19: "TOR",
        20: "VAN",
        21: "ATL",
        22: "MIN",
    }
    home_dict = {
        "364,1": "TOR",
        "364,0": "ATL",
        "365,1": "DC",
        "365,0": "NY",
        "366,1": "DAL",
        "366,0": "LA",
        "367,1": "HOU",
        "367,0": "CHI",
        "368,1": "MTL",
        "368,0": "NE",
        "369,1": "NYC",
        "369,0": "CLB",
        "370,1": "PHI",
        "370,0": "ORL",
        "371,1": "POR",
        "371,0": "VAN",
        "372,1": "RSL",
        "372,0": "SKC",
        "373,1": "SJ",
        "373,0": "MIN",
        "374,1": "SEA",
        "374,0": "COL",
    }

    fixture_data = json.load(
        open(
            data_utilities.get_raw_data_filepath([season_subdir, "Fixtures.json"]), "r"
        )
    )
    for entry in fixture_data:
        event_id = entry["id"]
        round_id = entry["event"]
        h_index = entry["team_h"]
        a_index = entry["team_a"]
        home_team = short_name_dict[h_index]
        away_team = short_name_dict[a_index]
        home_dict[str(event_id) + ",1"] = home_team
        home_dict[str(event_id) + ",0"] = away_team

    key_data = json.load(
        open(data_utilities.get_raw_data_filepath([season_subdir, "Key.json"]), "r")
    )
    for entry in key_data["elements"]:
        player_id = entry["id"]
        player_name = unidecode.unidecode(
            (entry["first_name"] + " " + entry["second_name"]).strip()
        )  # remove weird characters and extra space if player has one name
        position_id = entry["element_type"]

        player_data = json.load(
            open(
                data_utilities.get_raw_data_filepath(
                    [season_subdir, f"Player{player_id}.json"]
                ),
                "r",
            )
        )
        for player_entry in player_data["history"]:
            # points = player_entry['total_points']
            mins = player_entry["minutes"]
            if round_id not in saved_rounds and mins > 0:
                gls = player_entry["goals_scored"]
                ass = player_entry["assists"]
                cs = player_entry["clean_sheets"]
                sv = player_entry["saves"]
                pe = player_entry["penalties_earned"]
                ps = player_entry["penalties_saved"]
                pm = player_entry["penalties_missed"]
                gc = player_entry["goals_conceded"]
                yc = player_entry["yellow_cards"]
                rc = player_entry["red_cards"]
                og = player_entry["own_goals"]
                oga = player_entry["own_goal_earned"]
                sh = player_entry["shots"]
                wf = player_entry["was_fouled"]
                pss = player_entry["attempted_passes"]
                aps = player_entry["completed_passes"]
                pcp = aps / pss if pss > 0 else 0
                crs = player_entry["crosses"]
                kp = player_entry["key_passes"]
                bc = player_entry["big_chances_created"]
                cl = player_entry["clearances"]
                blk = player_entry["blocks"]
                intc = player_entry["interceptions"]
                tck = player_entry["tackles"]
                br = player_entry["recoveries"]
                elg = player_entry["errors_leading_to_goal"]
                cost = (
                    player_entry["value"] / 10
                )  # divide by 10 since costs are multiplied by 10 in raw json files to store it as an int instead of double

                round_id = player_entry["round"]
                event_id = player_entry["fixture"]
                home = int(player_entry["was_home"])
                opponent = short_name_dict[player_entry["opponent_team"]]
                team = home_dict[str(event_id) + "," + str(home)]

                player_dict = {
                    "mins": mins,
                    "gls": gls,
                    "ass": ass,
                    "cs": cs,
                    "sv": sv,
                    "pe": pe,
                    "ps": ps,
                    "pm": pm,
                    "gc": gc,
                    "yc": yc,
                    "rc": rc,
                    "og": og,
                    "oga": oga,
                    "sh": sh,
                    "wf": wf,
                    "pss": pss,
                    "aps": aps,
                    "pcp": pcp,
                    "crs": crs,
                    "kp": kp,
                    "bc": bc,
                    "cl": cl,
                    "blk": blk,
                    "intc": intc,
                    "tck": tck,
                    "br": br,
                    "elg": elg,
                    "position_id": position_id,
                    "player_id": player_id,
                    "player_name": player_name,
                    "team": team,
                    "round": round_id,
                    "event_id": event_id,
                    "opponent": opponent,
                    "home": home,
                    "season": season,
                }
                player_dict = data_utilities.fantasy_score(player_dict)
                rows_list.append(
                    tuple(
                        player_dict.get(colname, None)
                        for colname in [
                            col
                            for col in data_utilities.get_player_stats_columns().keys()
                        ]
                    )
                )
    if len(rows_list) > 0:
        query = Template(INSERT_TEMPLATE).render(n_columns=len(rows_list[0]),)
        cur = conn.connection.cursor()
        cur.executemany(query, rows_list)
        conn.connection.commit()
        cur.close()


@logging_utilities.instrument_function(logger)
def _insert_2018_season_data(conn, season):
    season_subdir = data_utilities.SEASON_DIRS[season]
    saved_rounds = get_saved_rounds(conn, season)

    historical_rows_list = []
    current_rows_list = []
    opponent_rows_list = []
    player_rows_list = []

    player_position_dict = {}
    player_team_dict = {}
    team_id_dict = {}
    player_name_dict = {}

    hit_current_round = False

    team_data = json.load(
        open(data_utilities.get_raw_data_filepath([season_subdir, "Squads.json"]), "r")
    )
    for team_entry in team_data:
        team_id = team_entry["id"]
        team_name = team_entry["short_name"]
        team_id_dict[team_id] = team_name

    player_data = json.load(
        open(data_utilities.get_raw_data_filepath([season_subdir, "Players.json"]), "r")
    )
    for player_entry in player_data:
        player_id = player_entry["id"]
        player_name = player_entry["known_name"]
        if player_name is None:
            player_name = unidecode.unidecode(
                (player_entry["first_name"] + " " + player_entry["last_name"]).strip()
            )  # remove weird characters and extra space if player has one name
        position_id = player_entry["positions"][0]
        current_cost = player_entry["cost"] / 1000000
        squad_id = player_entry["squad_id"]
        squad_name = team_id_dict[squad_id]
        player_dict = {
            "player_id": player_id,
            "player_name": player_name,
            "position_id": position_id,
            "cost": current_cost,
            "team": squad_name,
        }
        player_rows_list.append(player_dict)
        player_position_dict.update({player_id: position_id})
        player_team_dict.update({player_id: squad_name})
        player_name_dict.update({player_id: player_name})

    round_data = json.load(
        open(data_utilities.get_raw_data_filepath([season_subdir, "Rounds.json"]), "r")
    )
    for match_round in round_data:
        round_id = match_round["id"]
        if (
            round_id not in saved_rounds and match_round["status"] == "complete"
        ):  # or match_round['status'] == 'active':
            matches = match_round["matches"]
            for match in matches:
                match_id = match["id"]
                home_squad_id = match["home_squad_id"]
                away_squad_id = match["away_squad_id"]
                home_squad_short_name = match["home_squad_short_name"]
                away_squad_short_name = match["away_squad_short_name"]
                match_data = json.load(
                    open(
                        data_utilities.get_raw_data_filepath(
                            [season_subdir, "Matches", f"Match{match_id}.json"]
                        ),
                        "r",
                    )
                )
                for player_data in match_data:
                    player_id = player_data["player_id"]
                    player_entry = player_data["stats"]
                    mins = player_entry["MIN"]
                    if mins > 0:
                        team = player_team_dict.get(player_id)
                        position_id = player_position_dict.get(player_id)
                        player_name = player_name_dict.get(player_id)
                        gls = player_entry["GL"]
                        ass = player_entry["ASS"]
                        cs = player_entry["CS"]
                        sv = player_entry["SV"]
                        pe = player_entry["PE"]
                        ps = player_entry["PS"]
                        pm = player_entry["PM"]
                        gc = player_entry["GC"]
                        yc = player_entry["YC"]
                        rc = player_entry["RC"]
                        og = player_entry["OG"]
                        oga = player_entry["OGA"]
                        sh = player_entry["SH"]
                        wf = player_entry["WF"]
                        pss = player_entry["PSS"]
                        aps = player_entry["APS"]
                        pcp = aps / pss if pss > 0 else 0
                        crs = player_entry["CRS"]
                        kp = player_entry["KP"]
                        bc = player_entry["BC"]
                        cl = player_entry["CL"]
                        blk = player_entry["BLK"]
                        intc = player_entry["INT"]
                        tck = player_entry["TCK"]
                        br = player_entry["BR"]
                        elg = player_entry["ELG"]

                        if team == home_squad_short_name:
                            home = 1
                            team_id = home_squad_id
                            opponent_id = away_squad_id
                            team = home_squad_short_name
                            opponent = away_squad_short_name
                        elif team == away_squad_short_name:
                            home = 0
                            team_id = away_squad_id
                            opponent_id = home_squad_id
                            team = away_squad_short_name
                            opponent = home_squad_short_name
                        else:
                            home = 2
                            team_id = away_squad_id
                            opponent_id = home_squad_id
                            team = away_squad_short_name
                            opponent = home_squad_short_name

                        player_dict = {
                            "mins": mins,
                            "gls": gls,
                            "ass": ass,
                            "cs": cs,
                            "sv": sv,
                            "pe": pe,
                            "ps": ps,
                            "pm": pm,
                            "gc": gc,
                            "yc": yc,
                            "rc": rc,
                            "og": og,
                            "oga": oga,
                            "sh": sh,
                            "wf": wf,
                            "pss": pss,
                            "aps": aps,
                            "pcp": pcp,
                            "crs": crs,
                            "kp": kp,
                            "bc": bc,
                            "cl": cl,
                            "blk": blk,
                            "intc": intc,
                            "tck": tck,
                            "br": br,
                            "elg": elg,
                            "position_id": position_id,
                            "player_id": player_id,
                            "player_name": player_name,
                            "team": team,
                            "round": round_id,
                            "event_id": match_id,
                            "opponent": opponent,
                            "home": home,
                            "season": season,
                        }
                        player_dict = data_utilities.fantasy_score(player_dict)
                        historical_rows_list.append(player_dict)
        elif round_id not in saved_rounds and not hit_current_round:
            hit_current_round = True
            matches = match_round["matches"]
            for match in matches:
                match_id = match["id"]
                home_squad_id = match["home_squad_id"]
                away_squad_id = match["away_squad_id"]
                home_squad_short_name = match["home_squad_short_name"]
                away_squad_short_name = match["away_squad_short_name"]
                home_dict = {
                    "round": round_id,
                    "event_id": match_id,
                    "opponent": away_squad_short_name,
                    "team": home_squad_short_name,
                    "home": 1,
                    "season": season,
                }
                away_dict = {
                    "round": round_id,
                    "event_id": match_id,
                    "opponent": home_squad_short_name,
                    "team": away_squad_short_name,
                    "home": 0,
                    "season": season,
                }
                opponent_rows_list.append(home_dict)
                opponent_rows_list.append(away_dict)

    if len(historical_rows_list) > 0:
        df_historical = pd.DataFrame(historical_rows_list)
        # some home\away team and opponent info is messed up due to Fanhub only storing players current team so immidiately fix
        df_historical = df_historical.pipe(fix_player_home, player_team_dict)
        df_historical.to_sql("player_stats", conn, if_exists="append", index=False)
        conn.connection.commit()
    if len(opponent_rows_list) > 0:
        # add players to current round's dataframe by matching on current team
        df_player = pd.DataFrame(player_rows_list)
        df_opponent = pd.DataFrame(opponent_rows_list)
        df_current = pd.merge(df_player, df_opponent, how="right", on=["team"])
        df_current.to_sql("player_stats", conn, if_exists="append", index=False)
        conn.connection.commit()


@logging_utilities.instrument_function(logger)
def fix_player_home(df, player_team_dict):
    df_final = df.loc[df["home"] != 2].copy()
    df_player_transfered = df.loc[df["home"] == 2]
    df_player_counts = (
        df_player_transfered.groupby(["player_id"]).size().reset_index(name="count")
    )
    df_player_transfered = pd.merge(
        df_player_transfered, df_player_counts, how="left", on=["player_id"]
    )
    df_temp = df_player_transfered.loc[df_player_transfered["count"] == 1]
    df_temp.to_csv(
        data_utilities.get_processed_data_filepath("historical_player_anomalies"),
        sep=",",
        index=False,
    )
    df_player_trans_final = pd.DataFrame()
    df_player_transfered = df_player_transfered.loc[df_player_transfered["count"] >= 2]
    for player_id in df_player_transfered["player_id"].unique():
        df_subset = df.loc[(df["player_id"] == player_id)].copy()
        old_team = _get_old_team(df_subset)
        new_team = player_team_dict.get(player_id)
        if old_team != "unfound":
            round = _get_round_switch(df_subset, old_team, new_team)
            print(
                f"{player_id} switched from {old_team} to {new_team} in round {round}"
            )
            df_subset = df_subset.apply(
                _fill_correct_team, args=(old_team, round), axis=1
            )
            df_player_trans_final = pd.concat(
                [df_player_trans_final, df_subset], axis=0, sort=True
            )
            df_final = df_final.loc[df["player_id"] != player_id]
    if df_player_trans_final.shape[0] > 5:
        df_player_trans_final.to_csv(
            data_utilities.get_processed_data_filepath(
                "historical_player_anomalies_fixed"
            ),
            sep=",",
            index=False,
        )
    df_final = pd.concat([df_final, df_player_trans_final])
    return df_final


# helper function to correct 2018 player ids if they moved teams
def _get_old_team(df):
    team_list = []
    for index, row in df.iterrows():
        team = row["team"]
        opponent = row["opponent"]
        if team in team_list and opponent not in team_list:
            return team
        if opponent in team_list and team not in team_list:
            return opponent
        team_list.append(team)
        team_list.append(opponent)
    return "unfound"


# helper function to correct 2018 player ids if they moved teams
def _get_round_switch(df, old_team, new_team):
    for index, row in df.iterrows():
        team = row["team"]
        opponent = row["opponent"]
        if (new_team == team and old_team != opponent) or (
            new_team == opponent and old_team != team
        ):
            return row["round"]
    return 0


# helper function to correct 2018 player ids if they moved teams
def _fill_correct_team(row, old_team, round):
    if row["round"] < round:
        if row["team"] != old_team:
            row["opponent"] = row["team"]
            row["team"] = old_team
            row["home"] = 1
        else:
            row["home"] = 0
    return row


@logging_utilities.instrument_function(logger)
def _insert_monotonic_round_column(conn, season):
    cur = conn.connection.cursor()
    cur.execute(
        "SELECT MAX(unique_round) FROM player_stats WHERE season < ?;", (season,)
    )
    increment = cur.fetchone()[0]
    increment = increment if increment is not None else 0
    cur.execute(
        "UPDATE player_stats SET unique_round = round + ? WHERE season = ?;",
        (increment, season),
    )
    conn.connection.commit()
    cur.close()


@logging_utilities.instrument_function(logger)
def create_team_stats_view(conn):
    # no need for goals\clean sheets since it can be inferred from goals conceded, minutes will always be the same
    agg_dict = {
        stat_name: "SUM(" + stat_name + ")"
        for stat_name in list(
            set(data_utilities.all_feature_names())
            - set(["pcp", "cs", "gc", "gls", "mins"])
        )
    }
    agg_dict.update(
        {stat_name: "MAX(" + stat_name + ")" for stat_name in ["gc", "home"]}
    )
    agg_dict["pcp"] = "CAST(SUM(aps) AS FLOAT)/SUM(pss)"

    conn.execute("DROP VIEW IF EXISTS team_stats;")
    conn.connection.commit()

    template_args_dict = {"agg_dict": agg_dict}
    execute_templated_sql_script(conn, TEAM_STATS_VIEW_TEMPLATE, template_args_dict)


@logging_utilities.instrument_function(logger)
def create_name_matching_view(conn):
    player_id_match_dict = {
        36: 163042,  # Michael Azira
        71: 148995,  # Wil Trapp
        89: 211988,  # Chris Durkin
        133: 60214,  # A.J. DeLaGarza
        208: 67704,  # Antonio Delamea
        254: 220752,  # Derrick Etienne Jr.
        313: 45120,  # Ilson Pereira
        317: 95321,  # C.J. Sapong
        453: 37368,  # Michael Bradley
        459: 153647,  # Jonathan Osorio
        546: 196762,  # Fafa Picault
        557: 245005,  # Jake Nerwinski
        558: 227232,  # Artur de Lima Junior
        570: 204401,  # Michael Murillo
        618: 102921,  # Bernie Ibini
        632: 41004,  # Larrys Mabiala
        633: 77994,  # Marcelo Silva
        634: 88940,  # Valeri Qazaishvili
        644: 175077,  # Bill Tuiloma
        647: 92202,  # Pedro Santos
        655: 98898,  # Yoshi Yotun
    }

    conn.execute("DROP VIEW IF EXISTS name_matching;")
    conn.connection.commit()

    template_args_dict = {"player_id_match_dict": player_id_match_dict}
    execute_templated_sql_script(conn, NAME_MATCHING_VIEW_TEMPLATE, template_args_dict)


# TODO: turn this into a trigger
@logging_utilities.instrument_function(logger)
def create_train_test_split_table(conn, num_testing_rounds):
    conn.execute("DROP TABLE IF EXISTS train_test_split;")
    conn.execute("CREATE TABLE train_test_split(unique_round INT, dataset TEXT);")
    conn.connection.commit()

    template_args_dict = {"num_testing_rounds": num_testing_rounds}
    execute_templated_sql_script(conn, UPDATE_TRAIN_TEST_SPLIT, template_args_dict)


if __name__ == "__main__":
    parameters = config_utilities.get_parameter_dict(__file__)
    num_testing_rounds = parameters["num_testing_rounds"]
    refresh_views = parameters["refresh_views"]
    conn = data_utilities.initialize_db()
    create_player_stats_table(conn)
    insert_all_data(conn)
    if refresh_views:
        create_team_stats_view(conn)
        create_name_matching_view(conn)
    create_train_test_split_table(conn, num_testing_rounds)
    conn.close()
