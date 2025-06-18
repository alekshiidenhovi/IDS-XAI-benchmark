import time
import pandas as pd
import xgboost as xgb
import typing as T
import os
import click
import uuid
import datetime
from common.config import TrainingConfig, ParsedBaseTrainingKwargs
from datasets.unsw import UNSW_NB15
from common.utils import parse_int_list
from common.tracking import init_neptune_run


@click.command()
@click.option("--val-proportion", type=float, default=0.2)
@click.option("--dataset-shuffle", type=bool, default=True)
@click.option("--objective", type=str, default="multi:softmax")
@click.option("--step-log-interval", type=int, default=50)
@click.option("--random-state", type=int, default=42)
@click.option("--max-depths", callback=parse_int_list)
@click.option("--n-estimators", callback=parse_int_list)
@click.option("--min-child-weight", type=int, default=1)
@click.option("--learning-rate", type=float, default=0.01)
@click.option("--subsample", type=float, default=1.0)
@click.option("--colsample-bytree", type=float, default=1.0)
@click.option("--reg-alpha", type=float, default=0.0)
@click.option("--reg-lambda", type=float, default=0.0)
@click.option("--gamma", type=float, default=0.0)
def training_time_benchmark(**kwargs):
    benchmark_id = str(uuid.uuid4())

    parsed_kwargs = ParsedBaseTrainingKwargs(**kwargs)

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

    for max_depth in kwargs.get("max_depths"):
        for n_estimators in kwargs.get("n_estimators"):
            xgboost_params = {
                "objective": "multi:softmax",
                "max_depth": max_depth,
                "n_estimators": n_estimators,
                "min_child_weight": kwargs.get("min_child_weight"),
                "learning_rate": kwargs.get("learning_rate"),
                "subsample": kwargs.get("subsample"),
                "colsample_bytree": kwargs.get("colsample_bytree"),
                "reg_alpha": kwargs.get("reg_alpha"),
                "reg_lambda": kwargs.get("reg_lambda"),
                "gamma": kwargs.get("gamma"),
                "num_class": len(pd.unique(ytrain)),
            }

            training_config = TrainingConfig(
                **{**parsed_kwargs.model_dump(), **xgboost_params},
            )

            dtrain = xgb.DMatrix(Xtrain, label=ytrain)
            dval = xgb.DMatrix(Xvalidation, label=yvalidation)

            current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            experiment_name = f"{current_datetime}-training-time-benchmark-max_depth-{max_depth}-n_estimators-{n_estimators}"
            run = init_neptune_run(experiment_name)
            run["sys/tags"].add(["xgboost", "benchmark", "training_time"])
            for key, value in training_config.model_dump().items():
                run[f"config/{key}"] = value

            evals: T.List[T.Tuple[xgb.DMatrix, str]] = [
                (dtrain, "train"),
                (dval, "validation"),
            ]
            num_boost_round = xgboost_params.pop("n_estimators")

            training_start = time.time()
            model = xgb.train(
                params=xgboost_params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                evals=evals,
            )
            training_end = time.time()
            training_time = training_end - training_start

            local_model_path = (
                f".artifacts/{benchmark_id}/{experiment_name}/xgb_model.json"
            )
            os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
            model.save_model(local_model_path)
            run["artifacts/model"].upload(local_model_path)
            run["metrics/training_time"] = training_time

            run.stop()


if __name__ == "__main__":
    training_time_benchmark()
