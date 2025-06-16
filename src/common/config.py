from pydantic import BaseModel, Field
import typing as T
from common.tracking import init_wandb_api_client
from common.types import TRAINING_OBJECTIVE


class DatasetConfig(BaseModel):
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


class XGBoostClassifierConfig(BaseModel):
    """Configuration for the XGBoost Classifier."""

    objective: TRAINING_OBJECTIVE = Field(
        default="multi:softmax", description="Learning objective"
    )
    max_depth: int = Field(default=6, ge=1, description="Maximum depth of trees")
    min_child_weight: int = Field(
        default=1, ge=0, description="Minimum sum of instance weight needed in a child"
    )
    learning_rate: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Boosting learning rate"
    )
    subsample: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Subsample ratio of the training instances",
    )
    colsample_bytree: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Subsample ratio of columns when constructing each tree",
    )
    reg_alpha: float = Field(
        default=0.01, ge=0.0, description="L1 regularization term on weights"
    )
    reg_lambda: float = Field(
        default=1.0, ge=0.0, description="L2 regularization term on weights"
    )
    gamma: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum loss reduction required to make a further partition on a leaf node of the tree",
    )
    eval_metric: T.Literal["logloss", "mlogloss"] = Field(
        default="mlogloss", description="Evaluation metric"
    )
    num_class: int = Field(ge=2, description="Number of classes")


class OptimizationConfig(BaseModel):
    """Configuration for the optimization."""

    n_estimators: int = Field(
        default=100, ge=1, description="Number of boosting rounds"
    )
    step_log_interval: int = Field(
        default=10, ge=1, description="Interval at which to log metrics"
    )


class TrainingConfig(DatasetConfig, XGBoostClassifierConfig, OptimizationConfig):
    """Complete training configuration combining dataset, model and fine-tuning settings.

    Inherits from DatasetConfig, ModelConfig, FinetuningConfig and OptimizerConfig to provide a comprehensive configuration for the entire training pipeline.
    """

    random_state: int = Field(
        default=42,
        description="Seed for training reproducibility",
    )

    def get_dataset_config(self) -> DatasetConfig:
        """Get dataset-specific configuration."""
        return DatasetConfig(
            **{
                k: v
                for k, v in self.model_dump().items()
                if k in DatasetConfig.model_fields
            }
        )

    def get_xgboost_classifier_config(self) -> XGBoostClassifierConfig:
        """Get XGBoost Classifier configuration."""
        return XGBoostClassifierConfig(
            **{
                k: v
                for k, v in self.model_dump().items()
                if k in XGBoostClassifierConfig.model_fields
            }
        )

    def get_optimization_config(self) -> OptimizationConfig:
        """Get optimization configuration."""
        return OptimizationConfig(
            **{
                k: v
                for k, v in self.model_dump().items()
                if k in OptimizationConfig.model_fields
            }
        )

    @classmethod
    def load_from_wandb(cls, run_id: str):
        """Load the configuration from a W&B run."""
        wandb_api = init_wandb_api_client()
        run = wandb_api.run(run_id)
        valid_config = {k: v for k, v in run.config.items() if k in cls.model_fields}
        return cls(**valid_config)
