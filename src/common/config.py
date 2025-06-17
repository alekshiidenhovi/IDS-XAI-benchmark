from pydantic import BaseModel, Field
import typing as T
from common.tracking import init_wandb_api_client
from common.types import TRAINING_OBJECTIVE, EVAL_METRIC

T_Config = T.TypeVar("T_Config", bound="BaseConfig")


class BaseConfig(BaseModel):
    """Base configuration for all models."""


class DatasetConfig(BaseConfig):
    """Configuration for dataset loading and preprocessing.

    Contains parameters for data loading, batch sizes, image dimensions,
    dataset splits, and sampling configurations for training, validation and testing.
    """

    val_proportion: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Proportion of the training dataset to use for validation",
    )
    dataset_shuffle: bool = Field(
        default=True,
        description="Whether to shuffle the dataset",
    )


class XGBoostClassifierConfig(BaseConfig):
    """Configuration for the XGBoost Classifier."""

    objective: TRAINING_OBJECTIVE = Field(
        default="multi:softmax", description="Learning objective"
    )
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
    n_estimators: int = Field(
        default=100, ge=1, description="Number of boosting rounds"
    )

    @property
    def eval_metric(self) -> EVAL_METRIC:
        if self.objective == "multi:softmax":
            return "mlogloss"
        elif self.objective == "binary:logistic":
            return "logloss"
        else:
            raise ValueError(f"Invalid objective: {self.objective}")


class SettingsConfig(BaseConfig):
    """Configuration for the settings."""

    random_state: int = Field(
        default=42,
        description="Seed for training reproducibility",
    )
    step_log_interval: int = Field(
        default=10, ge=1, description="Interval at which to log metrics"
    )


class TrainingConfig(DatasetConfig, XGBoostClassifierConfig, SettingsConfig):
    """Complete training configuration combining dataset, model and fine-tuning settings.

    Inherits from DatasetConfig, XGBoostClassifierConfig and SettingsConfig to provide a comprehensive configuration for the entire training pipeline.
    """

    def get_config(self, config_class: T.Type[T_Config]) -> T_Config:
        """Get the configuration.

        Args:
            config_class: The configuration class to get the config for.

        Returns:
            An instance of the specified configuration class.
        """
        return config_class(
            **{
                k: v
                for k, v in self.model_dump().items()
                if k in config_class.model_fields
            }
        )

    @classmethod
    def load_from_wandb(cls, run_id: str):
        """Load the configuration from a W&B run."""
        wandb_api = init_wandb_api_client()
        run = wandb_api.run(run_id)
        valid_config = {k: v for k, v in run.config.items() if k in cls.model_fields}
        return cls(**valid_config)


def parse_valid_config_kwargs(config_kwargs: dict, config_class: BaseConfig) -> dict:
    valid_fields = set(config_class.model_fields.keys())
    return {
        k: v for k, v in config_kwargs.items() if v is not None and k in valid_fields
    }
