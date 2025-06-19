import time
import typing as T

import click
import optuna
import pandas as pd
import xgboost as xgb

from common.config import SettingsConfig, TrainingConfig
from common.metrics import OptimizationTimeMetrics, TrialMetrics
from common.pathing import get_local_optimization_metrics_file_path
from common.tracking import init_neptune_run
from common.utils import get_benchmark_id, get_current_datetime, parse_int_list
from datasets.unsw import UNSW_NB15
from storage.optimization_metrics import LocalOptimizationMetricsStorage


@click.command()
@click.option("--val-proportion", type=float, default=0.2)
@click.option("--dataset-shuffle", type=bool, default=True)
@click.option("--objective", type=str, default="multi:softmax")
@click.option("--step-log-interval", type=int, default=50)
@click.option("--random-state", type=int, default=42)
@click.option("--n-trials-list", callback=parse_int_list)
def optimize_xgboost(**kwargs):
    ## Benchmark parameters ##

    benchmark_id = get_benchmark_id()
    benchmark_name = "optimization-time-benchmark"

    ## Storage ##

    local_optimization_metrics_storage = LocalOptimizationMetricsStorage()

    ## Settings config ##

    settings_config = SettingsConfig(
        val_proportion=kwargs.get("val_proportion"),
        dataset_shuffle=kwargs.get("dataset_shuffle"),
        objective=kwargs.get("objective"),
        step_log_interval=kwargs.get("step_log_interval"),
        random_state=kwargs.get("random_state"),
    )

    ## Dataset ##

    train_dataset = UNSW_NB15(dataset_type="training")
    train_feature_matrix = train_dataset.get_feature_matrix()
    train_target_series = train_dataset.get_target_series(
        target_column_name=settings_config.target_column_name
    )
    Xtrain, Xvalidation, ytrain, yvalidation = train_dataset.create_train_val_splits(
        train_feature_matrix,
        train_target_series,
        val_proportion=settings_config.val_proportion,
        random_state=settings_config.random_state,
        dataset_shuffle=settings_config.dataset_shuffle,
    )

    ## Init logging ##
    n_trials_list = kwargs.get("n_trials_list")
    experiment_name = f"{get_current_datetime()}_n-trials-{','.join(str(n_trials) for n_trials in n_trials_list)}"
    run = init_neptune_run(experiment_name)
    run["sys/tags"].add(["xgboost", "benchmark", benchmark_name, "optuna"])

    best_validation_losses_per_study: T.List[float] = []
    total_execution_times_per_study: T.List[float] = []
    trials: T.List[TrialMetrics] = []

    for trial_idx, n_trials in enumerate(n_trials_list):
        validation_losses: T.List[float] = []
        execution_times: T.List[float] = []

        def optimize(trial):
            ## Training config ##

            xgboost_params = {
                "objective": settings_config.objective,
                "max_depth": trial.suggest_int("max_depth", 2, 12),
                "min_child_weight": trial.suggest_int("min_child_weight", 0, 3),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "num_class": len(pd.unique(ytrain)),
            }

            training_config = TrainingConfig(
                **{**settings_config.model_dump(), **xgboost_params},
            )

            ## Training ##

            dtrain = xgb.DMatrix(Xtrain, label=ytrain)
            dval = xgb.DMatrix(Xvalidation, label=yvalidation)

            evals: T.List[T.Tuple[xgb.DMatrix, str]] = [
                (dtrain, "train"),
                (dval, "validation"),
            ]
            evals_result = {}
            num_boost_round = xgboost_params.pop("n_estimators")

            training_start = time.perf_counter()
            model = xgb.train(
                params=xgboost_params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                evals=evals,
                evals_result=evals_result,
            )
            training_end = time.perf_counter()

            execution_time = training_end - training_start
            val_loss = evals_result["validation"][training_config.eval_metric][-1]

            validation_losses.append(val_loss)
            execution_times.append(execution_time)

            run.stop()

            return val_loss

        study = optuna.create_study(study_name="xgboost-training")
        study.optimize(optimize, n_trials=n_trials)

        ## Logging ##

        best_validation_losses_per_study.append(min(validation_losses))
        total_execution_times_per_study.append(sum(execution_times))

        trials.append(
            TrialMetrics(
                validation_losses=validation_losses,
                execution_times=execution_times,
            )
        )

    optimization_time_metrics = OptimizationTimeMetrics(
        best_validation_loss=min(best_validation_losses_per_study),
        best_execution_time=min(total_execution_times_per_study),
        n_trials_list=n_trials_list,
        best_validation_losses_per_study=best_validation_losses_per_study,
        total_execution_times_per_study=total_execution_times_per_study,
        trials=trials,
    )

    local_optimization_metrics_file_path = get_local_optimization_metrics_file_path(
        local_optimization_metrics_storage.dir_path, benchmark_id
    )
    local_optimization_metrics_storage.save_to_storage(
        optimization_time_metrics, local_optimization_metrics_file_path
    )

    run.stop()


if __name__ == "__main__":
    optimize_xgboost()
