import json
import os

import neptune
from pydantic import Field

from common.config import TrainingConfig
from storage.base_storage import ARTIFACTS_DIR, BaseStorage


class LocalTrainingConfigStorage(BaseStorage[TrainingConfig]):
    """
    Storage for the training config in local storage.
    """

    dir_path: str = Field(default=ARTIFACTS_DIR, description="Path to save config")
    file_name: str = Field(
        default="training_config.json", description="File name of saved config"
    )

    def save_to_storage(self, config: TrainingConfig, local_file_path: str) -> None:
        """Save training config to local JSON file."""
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        with open(local_file_path, "w") as f:
            json.dump(config.model_dump(), f, indent=2)

    @classmethod
    def load_from_storage(cls, local_file_path: str) -> TrainingConfig:
        """Load training config from local JSON file."""
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"Config file not found: {local_file_path}")

        with open(local_file_path, "r") as f:
            config_dict = json.load(f)

        return TrainingConfig(**config_dict)


class RemoteTrainingConfigStorage(BaseStorage[TrainingConfig]):
    """
    Storage for the training config in remote storage.
    """

    save_path_prefix: str = "config"

    def save_to_storage(self, run: neptune.Run, config: TrainingConfig) -> None:
        for field in TrainingConfig.model_fields:
            run[f"{self.save_path_prefix}/{field}"] = getattr(config, field)

    @classmethod
    def load_from_storage(cls, run: neptune.Run) -> TrainingConfig:
        try:
            config_dict = {}

            for field in TrainingConfig.model_fields:
                config_path = f"{cls.save_path_prefix}/{field}"
                try:
                    config_dict[field] = run[config_path].fetch()
                except Exception as e:
                    raise KeyError(
                        f"Missing expected config field in Neptune: {field}"
                    ) from e

            return TrainingConfig(**config_dict)

        except Exception as e:
            raise RuntimeError(f"Failed to load config from Neptune run {run._id}: {e}")
