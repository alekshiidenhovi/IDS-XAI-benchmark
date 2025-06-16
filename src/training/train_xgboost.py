import xgboost as xgb
import click
import datetime
import pandas as pd
from datasets.unsw import UNSW_NB15, multilabel_target_column
from common.tracking import init_wandb_run
from common.config import TrainingConfig
from training.callbacks import MetricsCallback
from common.types import TRAINING_OBJECTIVE


@click.command()
@click.option("--val-proportion", type=float, default=None)
@click.option("--dataset-shuffle", type=bool, default=None)
@click.option("--objective", type=TRAINING_OBJECTIVE, default=None)
@click.option("--max-depth", type=int, default=None)
@click.option("--min-child-weight", type=int, default=None)
@click.option("--learning-rate", type=float, default=None)
@click.option("--subsample", type=float, default=None)
@click.option("--colsample-bytree", type=float, default=None)
@click.option("--reg-alpha", type=float, default=None)
@click.option("--reg-lambda", type=float, default=None)
@click.option("--gamma", type=float, default=None)
@click.option("--eval-metric", type=str, default=None)
@click.option("--n-estimators", type=int, default=None)
@click.option("--step-log-interval", type=int, default=None)
@click.option("--random-state", type=int, default=None)
def train_xgboost_model(**kwargs):
    train_dataset = UNSW_NB15(dataset_type="training")
    train_feature_matrix = train_dataset.get_feature_matrix()
    train_target_series = train_dataset.get_target_series(
        target_column_name=multilabel_target_column
    )

    valid_fields = set(TrainingConfig.model_fields.keys())
    config_kwargs = {
        k: v for k, v in kwargs.items() if v is not None and k in valid_fields
    }
    training_config = TrainingConfig(
        **{**config_kwargs, "num_class": len(pd.unique(train_target_series))}
    )
    dataset_config = training_config.get_dataset_config()
    xgboost_config = training_config.get_xgboost_classifier_config()
    optimization_config = training_config.get_optimization_config()

    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_group_name = f"{current_datetime}-xgb_train-{optimization_config.n_estimators}_noftrees-{xgboost_config.max_depth}_depth-{xgboost_config.learning_rate}_lr-{xgboost_config.subsample}_subsample-{xgboost_config.colsample_bytree}_colsample"

    run = init_wandb_run(
        experiment_name="Experiment config",
        experiment_group_name=experiment_group_name,
        reinit=False,
    )
    run.config.update(training_config.model_dump())

    X_train, X_validation, y_train, y_validation = (
        train_dataset.create_train_val_splits(
            train_feature_matrix,
            train_target_series,
            val_proportion=dataset_config.val_proportion,
            random_state=training_config.random_state,
            dataset_shuffle=dataset_config.dataset_shuffle,
        )
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_validation, label=y_validation)
    eval_list = [(dtrain, "train"), (dval, "validation")]

    model = xgb.train(
        params=xgboost_config.model_dump(),
        dtrain=dtrain,
        num_boost_round=optimization_config.n_estimators,
        evals=eval_list,
        callbacks=[
            MetricsCallback(
                experiment_group_name,
                optimization_config.step_log_interval,
                dtrain,
                dval,
                y_train,
                y_validation,
            )
        ],
    )

    run.finish()


if __name__ == "__main__":
    train_xgboost_model()
