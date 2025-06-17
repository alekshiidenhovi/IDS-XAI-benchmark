import xgboost as xgb
import click
import pandas as pd
import datetime
import optuna
from datasets.unsw import UNSW_NB15, multilabel_target_column
from common.tracking import init_wandb_run
from common.utils import get_experiment_group_name
from common.config import (
    DatasetConfig,
    SettingsConfig,
    TrainingConfig,
    parse_valid_config_kwargs,
)
from common.hyperparams import get_xgboost_hyperparams
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
@click.option("--n-estimators", type=int, default=None)
@click.option("--step-log-interval", type=int, default=None)
@click.option("--random-state", type=int, default=None)
@click.option("--n-trials", type=int, default=200)
def train_xgboost_model(**kwargs):
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataset_config = DatasetConfig(**parse_valid_config_kwargs(kwargs, DatasetConfig))
    settings_config = SettingsConfig(
        **parse_valid_config_kwargs(kwargs, SettingsConfig)
    )

    train_dataset = UNSW_NB15(dataset_type="training")
    train_feature_matrix = train_dataset.get_feature_matrix()
    train_target_series = train_dataset.get_target_series(
        target_column_name=multilabel_target_column
    )

    X_train, X_validation, y_train, y_validation = (
        train_dataset.create_train_val_splits(
            train_feature_matrix,
            train_target_series,
            val_proportion=dataset_config.val_proportion,
            random_state=settings_config.random_state,
            dataset_shuffle=dataset_config.dataset_shuffle,
        )
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_validation, label=y_validation)

    def objective(trial: optuna.Trial) -> float:
        xgboost_config = get_xgboost_hyperparams(
            trial=trial,
            cli_kwargs=kwargs,
            num_classes=len(pd.unique(y_train)),
        )

        training_config = TrainingConfig(
            **dataset_config.model_dump(),
            **settings_config.model_dump(),
            **xgboost_config.model_dump(),
        )

        run = init_wandb_run(
            experiment_name="Experiment config",
            experiment_group_name=get_experiment_group_name(
                current_datetime=current_datetime,
                training_config=training_config,
            ),
            reinit=False,
        )

        run.config.update(training_config.model_dump())

        eval_list = [(dtrain, "train"), (dval, "validation")]
        evals_result = {}
        xgb.train(
            params=xgboost_config.model_dump(exclude={"n_estimators"}),
            dtrain=dtrain,
            num_boost_round=xgboost_config.n_estimators,
            evals=eval_list,
            evals_result=evals_result,
            callbacks=[
                MetricsCallback(
                    current_datetime=current_datetime,
                    training_config=training_config,
                    dtrain=dtrain,
                    dval=dval,
                    y_train=y_train,
                    y_validation=y_validation,
                )
            ],
        )

        run.finish()

        val_loss = evals_result["validation"][xgboost_config.eval_metric][-1]
        return val_loss

    study = optuna.create_study(study_name="xgboost-training")
    study.optimize(objective, n_trials=kwargs.get("n_trials"))


if __name__ == "__main__":
    train_xgboost_model()
