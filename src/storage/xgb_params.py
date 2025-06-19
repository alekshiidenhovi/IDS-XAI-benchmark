import json
import os

from pydantic import Field

from common.param_ranges import XGBoostParamRanges
from common.pathing import create_dir_if_not_exists
from storage.base_storage import ARTIFACTS_DIR, BaseStorage


class LocalXGBoostParamRangesStorage(BaseStorage[XGBoostParamRanges]):
    """
    Storage for the XGBoost param ranges in local storage.
    """

    dir_path: str = Field(default=ARTIFACTS_DIR, description="Path to save config")
    file_name: str = Field(
        default="xgb_param_ranges.json", description="File name of saved config"
    )

    def save_to_storage(self, config: XGBoostParamRanges, local_file_path: str) -> None:
        create_dir_if_not_exists(local_file_path)
        with open(local_file_path, "w") as f:
            json.dump(config.model_dump(), f, indent=2)

    @classmethod
    def load_from_storage(cls, local_file_path: str) -> XGBoostParamRanges:
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"Config file not found: {local_file_path}")

        with open(local_file_path, "r") as f:
            config_dict = json.load(f)

        return XGBoostParamRanges(**config_dict)
