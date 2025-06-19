import typing as T

from pydantic import BaseModel, Field

from common.types import EVAL_METRIC, TRAINING_OBJECTIVE
from datasets.unsw import binary_target_column, multilabel_target_column


class SHAPConfig(BaseModel):
    n_shap_background_samples: T.List[int] = Field(
        description="Number of background samples for SHAP"
    )
    n_shap_explained_samples: T.List[int] = Field(
        description="Number of explained samples for SHAP"
    )

    @classmethod
    def parse_kwargs(cls, **kwargs) -> "SHAPConfig":
        valid_fields = cls.model_fields.keys()
        parsed_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
        return cls.model_validate(parsed_kwargs)


class SettingsConfig(BaseModel):
    """
    Base settings for training.
    """

    val_proportion: float = Field(
        ge=0.0,
        le=1.0,
        description="Proportion of the training dataset to use for validation",
    )
    dataset_shuffle: bool = Field(
        description="Whether to shuffle the dataset",
    )

    objective: TRAINING_OBJECTIVE = Field(description="Learning objective")
    random_state: int = Field(
        description="Seed for training reproducibility",
    )
    step_log_interval: int = Field(ge=1, description="Interval at which to log metrics")

    @property
    def target_column_name(self) -> str:
        if self.objective == "multi:softmax":
            return multilabel_target_column
        elif self.objective == "binary:logistic":
            return binary_target_column
        else:
            raise ValueError(f"Invalid objective: {self.objective}")

    @classmethod
    def parse_kwargs(cls, **kwargs) -> "SettingsConfig":
        valid_fields = cls.model_fields.keys()
        parsed_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
        return cls.model_validate(parsed_kwargs)


class TrainingConfig(SettingsConfig):
    """Configuration for dataset loading and preprocessing.

    Contains parameters for data loading, batch sizes, image dimensions,
    dataset splits, and sampling configurations for training, validation and testing.
    """

    max_depth: int = Field(ge=1, description="Maximum depth of trees")
    min_child_weight: int = Field(
        ge=0, description="Minimum sum of instance weight needed in a child"
    )
    learning_rate: float = Field(ge=0.0, le=1.0, description="Boosting learning rate")
    subsample: float = Field(
        ge=0.0,
        le=1.0,
        description="Subsample ratio of the training instances",
    )
    colsample_bytree: float = Field(
        ge=0.0,
        le=1.0,
        description="Subsample ratio of columns when constructing each tree",
    )
    reg_alpha: float = Field(ge=0.0, description="L1 regularization term on weights")
    reg_lambda: float = Field(ge=0.0, description="L2 regularization term on weights")
    gamma: float = Field(
        ge=0.0,
        description="Minimum loss reduction required to make a further partition on a leaf node of the tree",
    )
    num_class: int = Field(ge=2, description="Number of classes")
    n_estimators: int = Field(ge=1, description="Number of boosting rounds")

    @property
    def eval_metric(self) -> EVAL_METRIC:
        if self.objective == "multi:softmax":
            return "mlogloss"
        elif self.objective == "binary:logistic":
            return "logloss"
        else:
            raise ValueError(f"Invalid objective: {self.objective}")

    @property
    def target_column_name(self) -> str:
        if self.objective == "multi:softmax":
            return multilabel_target_column
        elif self.objective == "binary:logistic":
            return binary_target_column
        else:
            raise ValueError(f"Invalid objective: {self.objective}")
