from joblib import dump

from src.models.logistic import LogisticModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel

from src.evaluation.metrics import evaluate