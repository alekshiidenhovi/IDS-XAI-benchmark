import optuna
import typing as T
from common.config import XGBoostClassifierConfig


def get_xgboost_hyperparams(
    trial: optuna.Trial, cli_kwargs: T.Dict[str, T.Any], num_classes: int
) -> XGBoostClassifierConfig:
    return XGBoostClassifierConfig(
        objective=cli_kwargs.get("objective") or "multi:softmax",
        max_depth=cli_kwargs.get("max_depth") or trial.suggest_int("max_depth", 2, 20),
        min_child_weight=cli_kwargs.get("min_child_weight")
        or trial.suggest_int("min_child_weight", 0, 3),
        learning_rate=cli_kwargs.get("learning_rate")
        or trial.suggest_float("learning_rate", 0.001, 0.1),
        subsample=cli_kwargs.get("subsample")
        or trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=cli_kwargs.get("colsample_bytree")
        or trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=cli_kwargs.get("reg_alpha")
        or trial.suggest_float("reg_alpha", 0.0, 1.0),
        reg_lambda=cli_kwargs.get("reg_lambda")
        or trial.suggest_float("reg_lambda", 0.0, 1.0),
        gamma=cli_kwargs.get("gamma") or trial.suggest_float("gamma", 0.0, 1.0),
        n_estimators=cli_kwargs.get("n_estimators")
        or trial.suggest_int("n_estimators", 400, 2000),
        num_class=num_classes,
    )
