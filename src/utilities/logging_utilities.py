import datetime as dt
import logging
import logging.config

from src.utilities import config_utilities, data_utilities

LOGGING_FILE = "logging.yaml"


def setup_logging():
    """
    Setup logging configuration. Only needs to be used in entry point scripts.
    Returns:
            None
    """
    config = config_utilities.parse_config(LOGGING_FILE)
    logging.config.dictConfig(config)


def instrument_function(logger):
    def real_instrument_function(func):
        def wrapper(*args, **kwargs):
            start_time = dt.datetime.now()
            result = func(*args, **kwargs)
            end_time = dt.datetime.now()
            logger.info(f"{func.__name__} took={end_time - start_time}")
            return result

        return wrapper

    return real_instrument_function


def instrument_pandas_piping_function(logger):
    def real_instrument_pandas_piping_function(func):
        def wrapper(df, *args, **kwargs):
            start_time = dt.datetime.now()
            result = func(df, *args, **kwargs)
            end_time = dt.datetime.now()
            logger.info(
                f"{func.__name__} took={end_time - start_time}, shape={result.shape}"
            )
            return result

        return wrapper

    return real_instrument_pandas_piping_function
