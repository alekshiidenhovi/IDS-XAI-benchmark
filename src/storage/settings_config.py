import json
import os

import neptune
from pydantic import Field

from common.config import SettingsConfig
from common.pathing import create_dir_if_not_exists
from storage.base_storage import ARTIFACTS_DIR, BaseStorage


class LocalSettingsConfigStorage(BaseStorage[SettingsConfig]):
    """
    Storage for the settings config in local storage.
    """

    dir_path: str = Field(default=ARTIFACTS_DIR, description="Path to save config")
    file_name: str = Field(
        default="settings_config.json", description="File name of saved config"
    )

    def save_to_storage(self, config: SettingsConfig, local_file_path: str) -> None:
        """Save settings config to local JSON file."""
        create_dir_if_not_exists(local_file_path)
        with open(local_file_path, "w") as f:
            json.dump(config.model_dump(), f, indent=2)

    @classmethod
    def load_from_storage(cls, local_file_path: str) -> SettingsConfig:
        """Load settings config from local JSON file."""
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"Config file not found: {local_file_path}")

        with open(local_file_path, "r") as f:
            config_dict = json.load(f)

        return SettingsConfig(**config_dict)


class RemoteSettingsConfigStorage(BaseStorage[SettingsConfig]):
    """
    Storage for the settings config in remote storage.
    """

    save_path_prefix: str = "config"

    def save_to_storage(self, run: neptune.Run, config: SettingsConfig) -> None:
        for field in SettingsConfig.model_fields:
            run[f"{self.save_path_prefix}/{field}"] = getattr(config, field)

    @classmethod
    def load_from_storage(cls, run: neptune.Run) -> SettingsConfig:
        try:
            config_dict = {}

            for field in SettingsConfig.model_fields:
                config_path = f"{cls.save_path_prefix}/{field}"
                try:
                    config_dict[field] = run[config_path].fetch()
                except Exception as e:
                    raise KeyError(
                        f"Missing expected config field in Neptune: {field}"
                    ) from e

            return SettingsConfig(**config_dict)

        except Exception as e:
            raise RuntimeError(f"Failed to load config from Neptune run {run._id}: {e}")
