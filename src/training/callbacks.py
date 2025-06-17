import numpy as np
import xgboost as xgb
import typing as T
from xgboost.callback import TrainingCallback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from neptune import Run
from common.config import TrainingConfig


class MetricsCallback(TrainingCallback):
    def __init__(
        self,
        run: Run,
        training_config: TrainingConfig,
        dtrain: xgb.DMatrix,
        dval: xgb.DMatrix,
        y_train: np.ndarray,
        y_validation: np.ndarray,
    ):
        self.run = run
        self.dtrain = dtrain
        self.dval = dval
        self.y_train = y_train
        self.y_validation = y_validation
        self.step_log_interval = training_config.step_log_interval

    def after_iteration(self, model: xgb.Booster, epoch: int, evals_log: T.Dict):
        if epoch % self.step_log_interval == 0:
            print(f"Logging metrics for iteration {epoch} to W&B")
            y_pred_train = model.predict(self.dtrain)
            y_pred_validation = model.predict(self.dval)

            train_accuracy = accuracy_score(self.y_train, y_pred_train)
            train_precision = precision_score(
                self.y_train, y_pred_train, average="weighted", zero_division=0
            )
            train_recall = recall_score(
                self.y_train, y_pred_train, average="weighted", zero_division=0
            )
            train_f1 = f1_score(
                self.y_train, y_pred_train, average="weighted", zero_division=0
            )

            val_accuracy = accuracy_score(self.y_validation, y_pred_validation)
            val_precision = precision_score(
                self.y_validation,
                y_pred_validation,
                average="weighted",
                zero_division=0,
            )
            val_recall = recall_score(
                self.y_validation,
                y_pred_validation,
                average="weighted",
                zero_division=0,
            )
            val_f1 = f1_score(
                self.y_validation,
                y_pred_validation,
                average="weighted",
                zero_division=0,
            )

            self.run["train/loss"].append(
                value=evals_log["train"]["mlogloss"][-1], step=epoch
            )
            self.run["train/accuracy"].append(value=train_accuracy, step=epoch)
            self.run["train/precision"].append(value=train_precision, step=epoch)
            self.run["train/recall"].append(value=train_recall, step=epoch)
            self.run["train/f1"].append(value=train_f1, step=epoch)
            self.run["validation/loss"].append(
                value=evals_log["validation"]["mlogloss"][-1], step=epoch
            )
            self.run["validation/accuracy"].append(value=val_accuracy, step=epoch)
            self.run["validation/precision"].append(value=val_precision, step=epoch)
            self.run["validation/recall"].append(value=val_recall, step=epoch)
            self.run["validation/f1"].append(value=val_f1, step=epoch)
