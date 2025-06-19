import typing as T

import numpy as np
import xgboost as xgb
from neptune import Run
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost.callback import TrainingCallback


class MetricsCallback(TrainingCallback):
    def __init__(
        self,
        run: Run,
        step_log_interval: int,
        dtrain: xgb.DMatrix,
        dval: xgb.DMatrix,
        ytrain: np.ndarray,
        yvalidation: np.ndarray,
    ):
        self.run = run
        self.dtrain = dtrain
        self.dval = dval
        self.ytrain = ytrain
        self.yvalidation = yvalidation
        self.step_log_interval = step_log_interval

    def after_iteration(self, model: xgb.Booster, epoch: int, evals_log: T.Dict):
        if epoch % self.step_log_interval == 0:
            print(f"Logging metrics for iteration {epoch}")
            y_pred_train = model.predict(self.dtrain)
            y_pred_validation = model.predict(self.dval)

            train_accuracy = accuracy_score(self.ytrain, y_pred_train)
            train_precision = precision_score(
                self.ytrain, y_pred_train, average="weighted", zero_division=0
            )
            train_recall = recall_score(
                self.ytrain, y_pred_train, average="weighted", zero_division=0
            )
            train_f1 = f1_score(
                self.ytrain, y_pred_train, average="weighted", zero_division=0
            )

            val_accuracy = accuracy_score(self.yvalidation, y_pred_validation)
            val_precision = precision_score(
                self.yvalidation,
                y_pred_validation,
                average="weighted",
                zero_division=0,
            )
            val_recall = recall_score(
                self.yvalidation,
                y_pred_validation,
                average="weighted",
                zero_division=0,
            )
            val_f1 = f1_score(
                self.yvalidation,
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
