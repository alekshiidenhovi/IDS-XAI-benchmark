import typing as T

import click
import pandas as pd
import xgboost as xgb

from common.config import ParsedBaseTrainingKwargs, TrainingConfig
from common.storage import TrainingConfigStorage
from common.tracking import init_neptune_run
from common.utils import get_benchmark_id
from datasets.unsw import UNSW_NB15
from training.callbacks import MetricsCallback


@click.command()
@click.option("--val-proportion", type=float, default=0.2)
@click.option("--dataset-shuffle", type=bool, default=True)
@click.option("--objective", type=str, default="multi:softmax")
@click.option("--step-log-interval", type=int, default=50)
@click.option("--random-state", type=int, default=42)
@click.option("--max-depth", type=int, default=12)
@click.option("--min-child-weight", type=int, default=1)
@click.option("--learning-rate", type=float, default=0.01)
@click.option("--subsample", type=float, default=1.0)
@click.option("--colsample-bytree", type=float, default=1.0)
@click.option("--reg-alpha", type=float, default=0.0)
@click.option("--reg-lambda", type=float, default=0.0)
@click.option("--gamma", type=float, default=0.0)
@click.option("--n-estimators", type=int, default=400)
def train_single_xgboost(**kwargs) -> float:
    run = init_neptune_run()

    parsed_kwargs = ParsedBaseTrainingKwargs.parse_kwargs(**kwargs)

    train_dataset = UNSW_NB15(dataset_type="training")
    train_feature_matrix = train_dataset.get_feature_matrix()
    train_target_series = train_dataset.get_target_series(
        target_column_name=parsed_kwargs.target_column_name
    )
    Xtrain, Xvalidation, ytrain, yvalidation = train_dataset.create_train_val_splits(
        train_feature_matrix,
        train_target_series,
        val_proportion=parsed_kwargs.val_proportion,
        random_state=parsed_kwargs.random_state,
        dataset_shuffle=parsed_kwargs.dataset_shuffle,
    )

    xgboost_params = {
        "objective": kwargs.get("objective"),
        "max_depth": kwargs.get("max_depth"),
        "min_child_weight": kwargs.get("min_child_weight"),
        "learning_rate": kwargs.get("learning_rate"),
        "subsample": kwargs.get("subsample"),
        "colsample_bytree": kwargs.get("colsample_bytree"),
        "reg_alpha": kwargs.get("reg_alpha"),
        "reg_lambda": kwargs.get("reg_lambda"),
        "gamma": kwargs.get("gamma"),
        "n_estimators": kwargs.get("n_estimators"),
        "num_class": len(pd.unique(ytrain)),
    }

    training_config = TrainingConfig(
        **{**kwargs, **xgboost_params},
    )

    training_config_storage = TrainingConfigStorage()
    training_config_storage.save_to_neptune(run, config=training_config)

    dtrain = xgb.DMatrix(Xtrain, label=ytrain)
    dval = xgb.DMatrix(Xvalidation, label=yvalidation)

    evals: T.List[T.Tuple[xgb.DMatrix, str]] = [(dtrain, "train"), (dval, "validation")]
    num_boost_round = xgboost_params.pop("n_estimators")
    model = xgb.train(
        params=xgboost_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        callbacks=[
            MetricsCallback(
                run=run,
                step_log_interval=training_config.step_log_interval,
                dtrain=dtrain,
                dval=dval,
                ytrain=ytrain,
                yvalidation=yvalidation,
            )
        ],
    )

    run.stop()


if __name__ == "__main__":
    train_single_xgboost()
