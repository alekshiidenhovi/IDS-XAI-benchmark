import time
import typing as T

import click
import optuna
import pandas as pd
import xgboost as xgb

from common.config import ParsedBaseTrainingKwargs, TrainingConfig
from common.storage import ModelStorage, TrainingConfigStorage
from common.tracking import init_neptune_run
from common.utils import get_benchmark_id, get_experiment_name
from datasets.unsw import UNSW_NB15
from training.callbacks import MetricsCallback


@click.command()
@click.option("--val-proportion", type=float, default=0.2)
@click.option("--dataset-shuffle", type=bool, default=True)
@click.option("--objective", type=str, default="multi:softmax")
@click.option("--step-log-interval", type=int, default=50)
@click.option("--random-state", type=int, default=42)
@click.option("--n-trials", type=int, default=200)
def optimize_xgboost(**kwargs):
    benchmark_id = get_benchmark_id()

    def optimize(trial):
        parsed_kwargs = ParsedBaseTrainingKwargs.parse_kwargs(**kwargs)

        train_dataset = UNSW_NB15(dataset_type="training")
        train_feature_matrix = train_dataset.get_feature_matrix()
        train_target_series = train_dataset.get_target_series(
            target_column_name=parsed_kwargs.target_column_name
        )
        Xtrain, Xvalidation, ytrain, yvalidation = (
            train_dataset.create_train_val_splits(
                train_feature_matrix,
                train_target_series,
                val_proportion=parsed_kwargs.val_proportion,
                random_state=parsed_kwargs.random_state,
                dataset_shuffle=parsed_kwargs.dataset_shuffle,
            )
        )

        xgboost_params = {
            "objective": kwargs.get("objective"),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_child_weight": trial.suggest_int("min_child_weight", 0, 3),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 400, 2000),
            "num_class": len(pd.unique(ytrain)),
        }

        training_config = TrainingConfig(
            **{**kwargs, **xgboost_params},
        )

        experiment_name = get_experiment_name(
            max_depth=training_config.max_depth,
            n_estimators=training_config.n_estimators,
        )
        run = init_neptune_run(experiment_name)
        run["sys/tags"].add(["xgboost", "optuna", "training"])

        training_config_storage = TrainingConfigStorage()
        model_storage = ModelStorage()

        training_config_storage.save_to_neptune(run, config=training_config)

        dtrain = xgb.DMatrix(Xtrain, label=ytrain)
        dval = xgb.DMatrix(Xvalidation, label=yvalidation)

        evals: T.List[T.Tuple[xgb.DMatrix, str]] = [
            (dtrain, "train"),
            (dval, "validation"),
        ]
        evals_result = {}
        num_boost_round = xgboost_params.pop("n_estimators")
        model = xgb.train(
            params=xgboost_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            evals_result=evals_result,
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

        model_storage.save_to_neptune(
            run, benchmark_id=benchmark_id, experiment_name=experiment_name, model=model
        )

        run.stop()

        val_loss = evals_result["validation"][training_config.eval_metric][-1]
        return val_loss

    study = optuna.create_study(study_name="xgboost-training")
    study.optimize(optimize, n_trials=kwargs.get("n_trials"))


if __name__ == "__main__":
    optimize_xgboost()
