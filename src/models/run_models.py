import logging
from typing import List

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
    run_models(model_classes)
