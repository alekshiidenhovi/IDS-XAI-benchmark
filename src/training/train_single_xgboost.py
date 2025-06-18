import xgboost as xgb
import pandas as pd
import typing as T
import shap
import numpy as np
import matplotlib.pyplot as plt
import click
from common.tracking import init_neptune_run
from common.config import TrainingConfig, ParsedBaseTrainingKwargs
from training.callbacks import MetricsCallback
from training.prepare_dataset import prepare_dataset


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

    Xtrain, Xvalidation, ytrain, yvalidation = prepare_dataset(
        target_column_name=parsed_kwargs.target_column_name,
        dataset_type="training",
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

    for key, value in training_config.model_dump().items():
        run[f"config/{key}"] = value

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

    N_SHAP_BG_SAMPLES = 200
    N_SHAP_EXPLAINED_SAMPLES = 100

    background_data = shap.kmeans(X=Xtrain, k=N_SHAP_BG_SAMPLES).data
    explainer = shap.TreeExplainer(
        model=model,
        data=background_data,
        feature_names=Xtrain.columns.tolist(),
        feature_perturbation="auto",
    )

    validation_indices = np.random.choice(
        len(Xvalidation), size=N_SHAP_EXPLAINED_SAMPLES, replace=False
    )
    validation_data = Xvalidation.iloc[validation_indices]

    shap_values = explainer.shap_values(validation_data)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, validation_data, feature_names=Xtrain.columns.tolist(), show=False
    )
    plt.tight_layout()
    plt.savefig("images/shap_summary.png", bbox_inches="tight", dpi=300)
    plt.close()
    run["explanations/shap_summary"].upload("images/shap_summary.png")

    run.stop()


if __name__ == "__main__":
    train_single_xgboost()
