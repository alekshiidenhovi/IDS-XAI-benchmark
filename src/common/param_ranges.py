import typing as T

from pydantic import BaseModel


class XGBoostParamRanges(BaseModel):
    max_depth: T.List[int]
    min_child_weight: T.List[int]
    learning_rate: T.List[float]
    subsample: T.List[float]
    colsample_bytree: T.List[float]
    reg_alpha: T.List[float]
    reg_lambda: T.List[float]
    gamma: T.List[float]
    n_estimators: T.List[int]
