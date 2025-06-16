import xgboost as xgb
import numpy as np
from datasets.unsw import UNSW_NB15, multilabel_target_column
from common.tracking import init_wandb_run
from training.callbacks import MetricsCallback


def train_xgboost_model():
    training_config = {
        "val_size": 0.2,
        "random_state": 42,
        "dataset_shuffle": True,
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.01,
        "reg_lambda": 1,
        "train_objective": "multi:softmax",
        "eval_metric": "mlogloss",
        "gamma": 0,
        "min_child_weight": 1,
        "step_log_interval": 10,
    }

    run = init_wandb_run(
        dataset_name="UNSW-NB15",
        experiment_name="xgboost_classification",
        experiment_group_name="model_training",
        reinit=True,
    )
    run.config.update(training_config)

    train_dataset = UNSW_NB15(dataset_type="training")
    train_feature_matrix = train_dataset.get_feature_matrix()
    train_target_series = train_dataset.get_target_series(
        target_column_name=multilabel_target_column
    )

    X_train, X_validation, y_train, y_validation = (
        train_dataset.create_train_val_splits(
            train_feature_matrix,
            train_target_series,
            val_size=training_config["val_size"],
            random_state=training_config["random_state"],
            dataset_shuffle=training_config["dataset_shuffle"],
        )
    )

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_validation, label=y_validation)

    # Create evaluation callback
    eval_list = [(dtrain, "train"), (dval, "validation")]

    # Train the model with evaluation sets
    model = xgb.train(
        params={
            "objective": training_config["train_objective"],
            "max_depth": training_config["max_depth"],
            "min_child_weight": training_config["min_child_weight"],
            "learning_rate": training_config["learning_rate"],
            "subsample": training_config["subsample"],
            "colsample_bytree": training_config["colsample_bytree"],
            "reg_alpha": training_config["reg_alpha"],
            "reg_lambda": training_config["reg_lambda"],
            "gamma": training_config["gamma"],
            "eval_metric": training_config["eval_metric"],
            "num_class": len(np.unique(y_train)),
        },
        dtrain=dtrain,
        num_boost_round=training_config["n_estimators"],
        evals=eval_list,
        callbacks=[
            MetricsCallback(run, training_config, dtrain, dval, y_train, y_validation)
        ],
    )

    run.finish()


if __name__ == "__main__":
    train_xgboost_model()
