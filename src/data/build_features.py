import logging

from jinja2 import Template

from src.data.make_database import execute_templated_sql_script
from src.utilities import config_utilities, data_utilities, logging_utilities

# For the SQLite LAG function to properly work update Python's SQLite as described in the link below
# https://dfir.blog/upgrading-pythons-sqlite/
PLAYER_LAGGING_STATS_VIEW_TEMPLATE = """
CREATE VIEW {{view_name}} AS
WITH player_stats_cte AS (
SELECT *
FROM player_stats
WHERE mins >= 45 OR unique_round == (SELECT MAX(unique_round) FROM player_stats))
SELECT player_stats_cte.player_id, player_stats_cte.season, round, event_id, {% for lag_amount in lag_offsets %}{% for lagged_stat in lagged_stats %}LAG({{lagged_stat}}, {{lag_amount}}, -1) OVER (PARTITION BY {% for partition in partitions %}{% if not loop.first %}, {% endif %}{{partition}}{% endfor %} ORDER BY unique_round) AS {{lagged_stat}}_lag_{{lag_amount}}_{{suffix}}{% if not loop.last %}, {% endif %}{% endfor %}{% if not loop.last %}, {% endif %}{% endfor %}
FROM player_stats_cte
LEFT JOIN advanced_position ON player_stats_cte.player_id = advanced_position.player_id AND player_stats_cte.season = advanced_position.season;
"""
TEAM_LAGGING_STATS_VIEW_TEMPLATE = """
CREATE VIEW {{view_name}} AS
SELECT event_id, team, opponent, {% for lag_amount in lag_offsets %}{% for lagged_stat in lagged_stats %}LAG({{lagged_stat}}, {{lag_amount}}, -1) OVER (PARTITION BY {% for partition in partitions %}{% if not loop.first %}, {% endif %}{{partition}}{% endfor %} ORDER BY unique_round) AS {{lagged_stat}}_lag_{{lag_amount}}_{{suffix}}{% if not loop.last %}, {% endif %}{% endfor %}{% if not loop.last %}, {% endif %}{% endfor %}
FROM team_stats;
"""
PLAYER_PERCENTILE_VIEW_TEMPLATE = """
CREATE VIEW advanced_position AS
WITH player_avg_stats_cte AS (
SELECT player_id, player_name, position_id, season, COUNT(*), {% for stat in stats_list %}AVG({{stat}}) AS {{stat}}{% if not loop.last %}, {% endif %}{% endfor %}
FROM player_stats
GROUP BY player_id, player_name, position_id, season),
player_perc_stats_cte AS (
SELECT
player_id, player_name, position_id, {% for stat in stats_list %}
    PERCENT_RANK() OVER( 
        PARTITION BY position_id ORDER BY {{stat}} 
    ) {{stat}}_percentile ,{% endfor %} season
FROM player_avg_stats_cte
)
SELECT *, CASE position_id
    WHEN 1 THEN 'GK'
    WHEN 2 THEN
    CASE
        WHEN (cl_percentile > 0.5 AND blk_percentile > 0.5) OR (cl_percentile > 0.8 OR blk_percentile > 0.8) THEN CASE
            WHEN (blk_percentile + cl_percentile - kp_percentile > 0.75) THEN 'CB'
            WHEN (blk_percentile + cl_percentile - kp_percentile > 0.5 AND pcp_percentile > 0.8) THEN 'CB'
            ELSE 'FB' END
        WHEN (kp_percentile > 0.95 OR sh_percentile > 0.95 OR bc_percentile > 0.9 OR crs_percentile > 0.95) AND crs_percentile > 0.5 AND (aps_percentile > 0.5 OR crs_percentile + bc_percentile > 1.8) THEN 'WB'
        ELSE 'FB' END
    WHEN 3 THEN
    CASE
        WHEN (tck_percentile + br_percentile + intc_percentile + blk_percentile > 3) AND (sh_percentile < 0.8) THEN CASE
            WHEN (sh_percentile < 0.7 OR pcp_percentile > 0.9) THEN 'DM' 
            ELSE 'CM' END
        WHEN (gls_percentile > 0.5 AND kp_percentile > 0.5) THEN CASE
            WHEN (bc_percentile + kp_percentile > 1.8) AND aps_percentile > 0.5 AND blk_percentile < 0.8 THEN 'AM'
            WHEN (aps_percentile > 0.75 OR crs_percentile < 0.25) THEN 'CM'
            ELSE 'WM' END
        WHEN (crs_percentile > 0.5 OR br_percentile < 0.25) AND (kp_percentile > 0.25) AND (pcp_percentile < 0.8) THEN 'WM'
        ELSE 'CM' END
    WHEN 4 THEN
    CASE
        WHEN (kp_percentile + bc_percentile > 1.75) THEN CASE
            WHEN (sh_percentile > 0.75 AND aps_percentile > 0.5) THEN 'CF'
            ELSE 'SS' END
        WHEN (sh_percentile < 0.5) AND (aps_percentile > 0.8 OR kp_percentile > 0.8) THEN 'SS'
        WHEN (br_percentile + intc_percentile + tck_percentile > 2.25 AND sh_percentile < 0.75) THEN 'SS'
        ELSE 'ST' END
    ELSE 'N/A' END advanced_position
FROM
player_perc_stats_cte;
"""


logging_utilities.setup_logging()
logger = logging.getLogger(__name__)


@logging_utilities.instrument_function(logger)
def create_player_percentile_view(conn):
    cur = conn.connection.cursor()
    cur.execute("DROP VIEW IF EXISTS advanced_position;")
    conn.connection.commit()

    template_args_dict = {"stats_list": data_utilities.important_stat_names()}
    execute_templated_sql_script(
        conn, PLAYER_PERCENTILE_VIEW_TEMPLATE, template_args_dict
    )

    conn.connection.commit()
    cur.close()


@logging_utilities.instrument_function(logger)
def create_player_lagging_stats_views(conn, max_lag_amount):
    lagged_stats = data_utilities.all_feature_names() + ["home"]
    lagging = [
        (
            "player_lagging_stats",
            ["player_stats_cte.player_id"],
            list(range(1, max_lag_amount)),
            "player",
        ),
        (
            "position_vs_opponent_lagging_stats",
            ["opponent", "advanced_position.advanced_position"],
            list(range(1, max_lag_amount)),
            "opp",
        ),
    ]
    cur = conn.connection.cursor()
    for view_name, partitions, lag_offsets, suffix in lagging:
        cur.execute(f"DROP VIEW IF EXISTS {view_name};")
        conn.connection.commit()

        template_args_dict = {
            "view_name": view_name,
            "partitions": partitions,
            "lag_offsets": lag_offsets,
            "lagged_stats": lagged_stats,
            "suffix": suffix,
        }
        execute_templated_sql_script(
            conn, PLAYER_LAGGING_STATS_VIEW_TEMPLATE, template_args_dict
        )

        conn.connection.commit()
    cur.close()


@logging_utilities.instrument_function(logger)
def create_team_lagging_stats_views(conn, max_lag_amount):
    lagged_stats = list(
        set(data_utilities.all_feature_names() + ["home", "dgw"])
        - set(["cs", "gls", "mins"])
    )
    lagged_stats = ["team_" + feature for feature in lagged_stats] + [
        "opp_" + feature for feature in lagged_stats
    ]
    lagging = [
        ("team_lagging_stats", ["team"], list(range(1, max_lag_amount)), "team_total"),
        (
            "opp_lagging_stats",
            ["opponent"],
            list(range(1, max_lag_amount)),
            "opp_total",
        ),
    ]
    cur = conn.connection.cursor()
    for view_name, partitions, lag_offsets, suffix in lagging:
        cur.execute(f"DROP VIEW IF EXISTS {view_name};")
        conn.connection.commit()

        template_args_dict = {
            "view_name": view_name,
            "partitions": partitions,
            "lag_offsets": lag_offsets,
            "lagged_stats": lagged_stats,
            "suffix": suffix,
        }
        execute_templated_sql_script(
            conn, TEAM_LAGGING_STATS_VIEW_TEMPLATE, template_args_dict
        )

        conn.connection.commit()
    cur.close()


if __name__ == "__main__":
    parameters = config_utilities.get_parameter_dict(__file__)
    max_lag_amount_player = parameters["max_lag_amount_player"]
    max_lag_amount_team = parameters["max_lag_amount_team"]
    conn = data_utilities.initialize_db()
    create_player_percentile_view(conn)
    create_player_lagging_stats_views(conn, max_lag_amount_player)
    create_team_lagging_stats_views(conn, max_lag_amount_team)
    conn.close()
