import json
import os

from pydantic import Field

from common.metrics import OptimizationTimeMetrics
from common.pathing import create_dir_if_not_exists
from storage.base_storage import ARTIFACTS_DIR, BaseStorage


class LocalOptimizationMetricsStorage(BaseStorage[OptimizationTimeMetrics]):
    """
    Storage for the optimization metrics in local storage.
    """

    dir_path: str = Field(default=ARTIFACTS_DIR, description="Path to save config")
    file_name: str = Field(
        default="optimization_metrics.json", description="File name of saved config"
    )

    def save_to_storage(
        self, metrics: OptimizationTimeMetrics, local_file_path: str
    ) -> None:
        """Save optimization metrics to local JSON file."""
        create_dir_if_not_exists(local_file_path)
        with open(local_file_path, "w") as f:
            json.dump(metrics.model_dump(), f, indent=2)

    @classmethod
    def load_from_storage(cls, local_file_path: str) -> OptimizationTimeMetrics:
        """Load optimization metrics from local JSON file."""
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"Config file not found: {local_file_path}")

        with open(local_file_path, "r") as f:
            config_dict = json.load(f)

        return OptimizationTimeMetrics(**config_dict)
