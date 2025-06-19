import time
import typing as T

import click
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from common.config import SettingsConfig, SHAPConfig, TrainingConfig
from common.pathing import (
    create_dir_if_not_exists,
    get_explanation_time_metrics_file_path,
    get_local_model_file_path,
    get_local_settings_config_file_path,
    get_local_shap_config_file_path,
    get_local_training_config_file_path,
    get_training_time_metrics_file_path,
)
from common.tracking import init_neptune_run
from common.utils import get_benchmark_id, get_current_datetime, parse_int_list
from datasets.unsw import UNSW_NB15
from storage.model import LocalModelStorage
from storage.settings_config import LocalSettingsConfigStorage
from storage.shap_config import LocalSHAPConfigStorage, RemoteSHAPConfigStorage
from storage.training_config import (
    LocalTrainingConfigStorage,
    RemoteTrainingConfigStorage,
)
from training.callbacks import MetricsCallback


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
@click.option("--n-shap-background-samples", callback=parse_int_list)
@click.option("--n-shap-explained-samples", callback=parse_int_list)
def training_time_benchmark(**kwargs):
    ## Benchmark parameters ##

    benchmark_id = get_benchmark_id()
    benchmark_name = "training-time-benchmark"

    ## Storage ##

    local_training_config_storage = LocalTrainingConfigStorage()
    local_model_storage = LocalModelStorage()
    local_settings_config_storage = LocalSettingsConfigStorage()
    local_shap_config_storage = LocalSHAPConfigStorage()

    remote_training_config_storage = RemoteTrainingConfigStorage()
    remote_shap_config_storage = RemoteSHAPConfigStorage()

    ## Settings config ##

    settings_config = SettingsConfig.parse_kwargs(**kwargs)

    local_settings_config_file_path = get_local_settings_config_file_path(
        local_settings_config_storage, benchmark_id
    )
    local_settings_config_storage.save_to_storage(
        config=settings_config,
        local_file_path=local_settings_config_file_path,
    )

    ## SHAP config ##

    shap_config = SHAPConfig.parse_kwargs(**kwargs)

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

    training_time_metrics: T.List[T.Dict[str, T.Any]] = []
    explanation_time_metrics: T.List[T.Dict[str, T.Any]] = []

    for max_depth in kwargs.get("max_depths"):
        for n_estimators in kwargs.get("n_estimators"):
            ## Training config ##

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
                **{**settings_config.model_dump(), **xgboost_params},
            )

            ## Init logging ##

            experiment_name = f"{get_current_datetime()}_max-depth-{max_depth}_n-estimators-{n_estimators}"
            run = init_neptune_run(experiment_name)
            run["sys/tags"].add(["xgboost", "benchmark", benchmark_name])

            ## Training ##

            dtrain = xgb.DMatrix(Xtrain, label=ytrain)
            dval = xgb.DMatrix(Xvalidation, label=yvalidation)

            evals: T.List[T.Tuple[xgb.DMatrix, str]] = [
                (dtrain, "train"),
                (dval, "validation"),
            ]
            num_boost_round = xgboost_params.pop("n_estimators")
            evals_result = {}

            training_start = time.perf_counter()
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
            training_end = time.perf_counter()
            training_time = training_end - training_start
            print(f"Training time: {training_time} seconds")

            ## Explanation ##
            for N_SHAP_BACKGROUND_SAMPLES in shap_config.n_shap_background_samples:
                for N_SHAP_EXPLAINED_SAMPLES in shap_config.n_shap_explained_samples:
                    print(
                        f"Max_depth: {max_depth}, N_estimators: {n_estimators}, N_SHAP_BACKGROUND_SAMPLES: {N_SHAP_BACKGROUND_SAMPLES}, N_SHAP_EXPLAINED_SAMPLES: {N_SHAP_EXPLAINED_SAMPLES}",
                        end=" ||| ",
                        flush=True,
                    )
                    explanation_start = time.perf_counter()
                    background_data = shap.kmeans(
                        Xtrain, N_SHAP_BACKGROUND_SAMPLES
                    ).data
                    explainer = shap.TreeExplainer(
                        model=model,
                        data=background_data,
                        feature_names=Xtrain.columns.tolist(),
                        feature_perturbation="auto",
                    )
                    validation_indices = np.random.choice(
                        len(Xvalidation), size=N_SHAP_EXPLAINED_SAMPLES, replace=False
                    )
                    x_val_samples = Xvalidation.iloc[validation_indices]
                    y_val_samples = yvalidation.iloc[validation_indices]
                    shap_values = explainer.shap_values(x_val_samples, y_val_samples)
                    explanation_end = time.perf_counter()
                    explanation_time = explanation_end - explanation_start
                    print(f"Explanation time: {explanation_time} seconds")
                    explanation_time_metrics.append(
                        {
                            "experiment_name": experiment_name,
                            "max_depth": max_depth,
                            "n_estimators": n_estimators,
                            "n_shap_background_samples": N_SHAP_BACKGROUND_SAMPLES,
                            "n_shap_explained_samples": N_SHAP_EXPLAINED_SAMPLES,
                            "explanation_time": explanation_time,
                        }
                    )

            ## Logging ##

            ### Training config ###

            local_training_config_file_path = get_local_training_config_file_path(
                dir_path=local_training_config_storage.dir_path,
                benchmark_name=benchmark_name,
                experiment_name=experiment_name,
                benchmark_id=benchmark_id,
                file_name=local_training_config_storage.file_name,
            )
            local_training_config_storage.save_to_storage(
                config=training_config,
                local_file_path=local_training_config_file_path,
            )
            remote_training_config_storage.save_to_storage(
                run,
                training_config,
            )

            ## SHAP config ##
            local_shap_config_file_path = get_local_shap_config_file_path(
                dir_path=local_shap_config_storage.dir_path,
                benchmark_name=benchmark_name,
                experiment_name=experiment_name,
                benchmark_id=benchmark_id,
                file_name=local_shap_config_storage.file_name,
            )
            local_shap_config_storage.save_to_storage(
                config=shap_config,
                local_file_path=local_shap_config_file_path,
            )

            ### Model ###

            local_model_file_path = get_local_model_file_path(
                dir_path=local_model_storage.dir_path,
                benchmark_name=benchmark_name,
                experiment_name=experiment_name,
                benchmark_id=benchmark_id,
                file_name=local_model_storage.file_name,
            )
            local_model_storage.save_to_storage(model, local_model_file_path)

            ### Metrics ###

            val_loss = evals_result["validation"][training_config.eval_metric][-1]
            val_f1 = evals_result["validation"]["f1"][-1]

            run["metrics/execution_time"] = training_time
            run["metrics/validation_loss"] = val_loss
            run["metrics/validation_f1"] = val_f1

            run.stop()

            training_time_metrics.append(
                {
                    "experiment_name": experiment_name,
                    "max_depth": max_depth,
                    "n_estimators": n_estimators,
                    "val_loss": val_loss,
                    "execution_time": training_time,
                }
            )

    ## Save metrics ##

    training_time_dir_path = get_training_time_metrics_file_path(
        dir_path=local_training_config_storage.dir_path,
        benchmark_name=benchmark_name,
        benchmark_id=benchmark_id,
    )
    create_dir_if_not_exists(training_time_dir_path)
    training_time_df = pd.DataFrame(training_time_metrics)
    training_time_df.to_parquet(training_time_dir_path)

    explanation_time_dir_path = get_explanation_time_metrics_file_path(
        dir_path=local_training_config_storage.dir_path,
        benchmark_name=benchmark_name,
        benchmark_id=benchmark_id,
    )
    create_dir_if_not_exists(explanation_time_dir_path)
    explanation_time_df = pd.DataFrame(explanation_time_metrics)
    explanation_time_df.to_parquet(explanation_time_dir_path)


if __name__ == "__main__":
    training_time_benchmark()
