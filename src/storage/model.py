import json
import os

import neptune
import xgboost as xgb
from pydantic import Field

from common.pathing import create_dir_if_not_exists
from storage.base_storage import ARTIFACTS_DIR, BaseStorage


class LocalModelStorage(BaseStorage[xgb.Booster]):
    """
    Storage for the training config in local storage.
    """

    dir_path: str = Field(default=ARTIFACTS_DIR, description="Path to save model")
    file_name: str = Field(
        default="xgb_model.json", description="File name of saved model"
    )

    def save_to_storage(self, model: xgb.Booster, local_file_path: str) -> None:
        create_dir_if_not_exists(local_file_path)
        model.save_model(local_file_path)

    @classmethod
    def load_from_storage(cls, local_file_path: str) -> xgb.Booster:
        """Load training config from local JSON file."""
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"Model file not found: {local_file_path}")

        model = xgb.Booster()
        model.load_model(local_file_path)
        return model


class RemoteModelStorage(BaseStorage[xgb.Booster]):
    """
    Storage for the training config in remote storage.
    """

    save_path_prefix: str = "artifacts/model"
    file_name: str = "xgb_model.json"

    def save_to_storage(self, run: neptune.Run, local_model_path: str) -> None:
        run[self.save_path_prefix].upload(local_model_path)

    @classmethod
    def load_from_storage(
        cls, run: neptune.Run, local_model_file_path: str
    ) -> xgb.Booster:
        run[cls.save_path_prefix].download(local_model_file_path)
        return LocalModelStorage.load_from_storage(local_model_file_path)
