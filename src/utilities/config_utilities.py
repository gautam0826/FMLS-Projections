import argparse
import os
import sys
from typing import Any, Dict

import yaml

from src.utilities import data_utilities

PARAMETERS_FILE = "parameters.yaml"


def get_parameter_dict(file: str) -> Dict[str, Any]:
    config_parameters = get_config_parameter_dict(file)
    parser = argparse.ArgumentParser()
    for parameter, value in config_parameters.items():
        parser.add_argument(f"--{parameter}", type=type(value), default=value)
    args = parser.parse_args(sys.argv[1:])
    return vars(args)


def get_config_parameter_dict(file: str) -> Dict[str, Any]:
    parameter_key = os.path.basename(file.replace(".py", "")) + "_params"
    return parse_config(PARAMETERS_FILE)[parameter_key]


def parse_config(config_file: str) -> Dict[str, Any]:
    """
	Helper function to parse the Yaml config files.
	Args:
		config_file (str): yaml configuration file
	Returns:
		config: parsed dictionary object
	"""
    with open(data_utilities.get_conf_file_path(config_file), "r") as f:
        config = yaml.safe_load(f)
    return config
