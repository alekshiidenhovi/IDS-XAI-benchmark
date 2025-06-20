import time
import typing as T

import click
import optuna
import optuna.integration.xgboost
import pandas as pd
import xgboost as xgb

from common.config import SettingsConfig, TrainingConfig
from common.metrics import OptimizationTimeMetrics, TrialMetrics
from common.param_ranges import XGBoostParamRanges
from common.pathing import (
    get_local_model_file_path,
    get_local_optimization_metrics_file_path,
    get_local_xgb_param_ranges_file_path,
)
from common.tracking import init_neptune_run
from common.utils import get_benchmark_id, get_current_datetime, parse_int_list
from datasets.unsw import UNSW_NB15
from storage.model import LocalModelStorage
from storage.optimization_metrics import LocalOptimizationMetricsStorage
from storage.xgb_params import LocalXGBoostParamRangesStorage


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
    local_xgb_param_ranges_storage = LocalXGBoostParamRangesStorage()
    local_model_storage = LocalModelStorage()

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

    xgb_param_ranges = XGBoostParamRanges(
        max_depth=[2, 6],
        min_child_weight=[0, 3],
        learning_rate=[0.001, 0.1],
        subsample=[0.6, 1.0],
        colsample_bytree=[0.6, 1.0],
        reg_alpha=[0.0, 1.0],
        reg_lambda=[0.0, 1.0],
        gamma=[0.0, 1.0],
        n_estimators=[100, 500],
    )

    for trial_idx, n_trials in enumerate(n_trials_list):
        validation_losses: T.List[float] = []
        execution_times: T.List[float] = []

        def optimize(trial: optuna.Trial):
            ## Training config ##

            xgboost_params = {
                "objective": settings_config.objective,
                "max_depth": trial.suggest_int(
                    "max_depth",
                    xgb_param_ranges.max_depth[0],
                    xgb_param_ranges.max_depth[1],
                ),
                "min_child_weight": trial.suggest_int(
                    "min_child_weight",
                    xgb_param_ranges.min_child_weight[0],
                    xgb_param_ranges.min_child_weight[1],
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    xgb_param_ranges.learning_rate[0],
                    xgb_param_ranges.learning_rate[1],
                ),
                "subsample": trial.suggest_float(
                    "subsample",
                    xgb_param_ranges.subsample[0],
                    xgb_param_ranges.subsample[1],
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree",
                    xgb_param_ranges.colsample_bytree[0],
                    xgb_param_ranges.colsample_bytree[1],
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha",
                    xgb_param_ranges.reg_alpha[0],
                    xgb_param_ranges.reg_alpha[1],
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda",
                    xgb_param_ranges.reg_lambda[0],
                    xgb_param_ranges.reg_lambda[1],
                ),
                "gamma": trial.suggest_float(
                    "gamma", xgb_param_ranges.gamma[0], xgb_param_ranges.gamma[1]
                ),
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    xgb_param_ranges.n_estimators[0],
                    xgb_param_ranges.n_estimators[1],
                ),
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

            pruning_callback = optuna.integration.XGBoostPruningCallback(
                trial, f"validation-{training_config.eval_metric}"
            )

            training_start = time.perf_counter()
            model = xgb.train(
                params=xgboost_params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                evals=evals,
                evals_result=evals_result,
                callbacks=[pruning_callback],
            )
            training_end = time.perf_counter()

            ## Model ##

            local_model_file_path = get_local_model_file_path(
                dir_path=local_model_storage.dir_path,
                benchmark_name=benchmark_name,
                experiment_name=f"{n_trials}-{trial.number}",
                benchmark_id=benchmark_id,
                file_name=local_model_storage.file_name,
            )
            local_model_storage.save_to_storage(model, local_model_file_path)

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
        dir_path=local_optimization_metrics_storage.dir_path,
        benchmark_name=benchmark_name,
        benchmark_id=benchmark_id,
    )
    local_optimization_metrics_storage.save_to_storage(
        optimization_time_metrics, local_optimization_metrics_file_path
    )

    local_xgb_param_ranges_file_path = get_local_xgb_param_ranges_file_path(
        dir_path=local_xgb_param_ranges_storage.dir_path,
        benchmark_name=benchmark_name,
        benchmark_id=benchmark_id,
        file_name=local_xgb_param_ranges_storage.file_name,
    )
    local_xgb_param_ranges_storage.save_to_storage(
        xgb_param_ranges, local_xgb_param_ranges_file_path
    )

    run.stop()


if __name__ == "__main__":
    optimize_xgboost()
