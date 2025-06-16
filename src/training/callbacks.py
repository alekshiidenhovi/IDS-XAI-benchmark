import numpy as np
import xgboost as xgb
import typing as T
from xgboost.callback import TrainingCallback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from common.tracking import init_wandb_run


class MetricsCallback(TrainingCallback):
    def __init__(
        self,
        experiment_group_name: str,
        step_log_interval: int,
        dtrain: xgb.DMatrix,
        dval: xgb.DMatrix,
        y_train: np.ndarray,
        y_validation: np.ndarray,
    ):
        self.run = init_wandb_run(
            experiment_name="Training metrics",
            experiment_group_name=experiment_group_name,
            reinit=True,
        )
        self.step_log_interval = step_log_interval
        self.dtrain = dtrain
        self.dval = dval
        self.y_train = y_train
        self.y_validation = y_validation

    def after_iteration(self, model: xgb.Booster, epoch: int, evals_log: T.Dict):
        if epoch % self.step_log_interval == 0:
            print(f"Logging metrics for iteration {epoch}")
            y_pred_train = model.predict(self.dtrain)
            y_pred_validation = model.predict(self.dval)

            train_accuracy = accuracy_score(self.y_train, y_pred_train)
            train_precision = precision_score(
                self.y_train, y_pred_train, average="weighted"
            )
            train_recall = recall_score(self.y_train, y_pred_train, average="weighted")
            train_f1 = f1_score(self.y_train, y_pred_train, average="weighted")

            val_accuracy = accuracy_score(self.y_validation, y_pred_validation)
            val_precision = precision_score(
                self.y_validation, y_pred_validation, average="weighted"
            )
            val_recall = recall_score(
                self.y_validation, y_pred_validation, average="weighted"
            )
            val_f1 = f1_score(self.y_validation, y_pred_validation, average="weighted")

            self.run.log(
                {
                    "iteration": epoch,
                    "train_loss": evals_log["train"]["mlogloss"][-1],
                    "train_accuracy": train_accuracy,
                    "train_precision": train_precision,
                    "train_recall": train_recall,
                    "train_f1": train_f1,
                    "val_loss": evals_log["validation"]["mlogloss"][-1],
                    "val_accuracy": val_accuracy,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                }
            )
