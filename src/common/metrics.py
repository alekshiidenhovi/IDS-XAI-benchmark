import typing as T

from pydantic import BaseModel, Field


class TrialMetrics(BaseModel):
    validation_losses: T.List[float] = Field(description="Validation losses")
    execution_times: T.List[float] = Field(description="Execution times")


class OptimizationTimeMetrics(BaseModel):
    best_validation_loss: float = Field(description="Best validation loss")
    best_execution_time: float = Field(description="Best execution times per trial")
    n_trials_list: T.List[int] = Field(description="List of number of trials")
    best_validation_losses_per_study: T.List[float] = Field(
        description="Best validation losses per study"
    )
    total_execution_times_per_study: T.List[float] = Field(
        description="Total execution times per study"
    )
    trials: T.List[TrialMetrics] = Field(description="Trials")
