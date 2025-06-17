import xgboost as xgb
import pandas as pd
import typing as T
import click
from datasets.unsw import multilabel_target_column, binary_target_column
from common.tracking import init_neptune_run
from common.config import TrainingConfig
from training.prepare_dataset import prepare_dataset
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

    if kwargs.get("objective") == "multi:softmax":
        target_column_name = multilabel_target_column
    elif kwargs.get("objective") == "binary:logistic":
        target_column_name = binary_target_column
    else:
        raise ValueError(f"Invalid objective: {kwargs.get('objective')}")

    Xtrain, Xvalidation, ytrain, yvalidation = prepare_dataset(
        target_column_name=target_column_name,
        dataset_type="training",
        val_proportion=kwargs.get("val_proportion"),
        random_state=kwargs.get("random_state"),
        dataset_shuffle=kwargs.get("dataset_shuffle"),
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
        "num_class": len(pd.unique(ytrain)),
    }

    training_config = TrainingConfig(
        **{**kwargs, **xgboost_params},
    )

    for key, value in training_config.model_dump().items():
        run[f"config/{key}"] = value

    dtrain = xgb.DMatrix(Xtrain, label=ytrain)
    dval = xgb.DMatrix(Xvalidation, label=yvalidation)

    evals: T.List[T.Tuple[xgb.DMatrix, str]] = [(dtrain, "train"), (dval, "validation")]
    xgb.train(
        params=xgboost_params,
        dtrain=dtrain,
        num_boost_round=training_config.n_estimators,
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
