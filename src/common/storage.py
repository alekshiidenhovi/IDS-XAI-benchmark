import os
from typing import Generic, Type, TypeVar

import neptune
import xgboost as xgb
from pydantic import BaseModel

from common.config import TrainingConfig

T = TypeVar("T", bound="NeptuneStorage")


class NeptuneStorage(BaseModel, Generic[T]):
    """
    Abstract base class for Neptune storage.
    """

    class Config:
        arbitrary_types_allowed = True

    def save_to_neptune(self, run: neptune.Run, **kwargs) -> None:
        raise NotImplementedError

    @classmethod
    def load_from_neptune(cls: Type[T], run: neptune.Run, **kwargs) -> T:
        raise NotImplementedError


class ModelStorage(NeptuneStorage[xgb.Booster]):
    """
    Storage for the artifact model.
    """

    neptune_save_path: str = "artifacts/model"
    file_name: str = "xgb_model.json"

    @staticmethod
    def local_dir_path(benchmark_id: str, experiment_name: str) -> str:
        return f".artifacts/{benchmark_id}/{experiment_name}"

    def save_to_neptune(self, run: neptune.Run, **kwargs) -> None:
        if "benchmark_id" in kwargs:
            benchmark_id: str = kwargs["benchmark_id"]
        else:
            raise ValueError("benchmark_id is required")

        if "experiment_name" in kwargs:
            experiment_name: str = kwargs["experiment_name"]
        else:
            raise ValueError("experiment_name is required")

        if "model" in kwargs:
            model: xgb.Booster = kwargs["model"]
        else:
            raise ValueError("model is required")

        local_model_path = (
            f"{self.local_dir_path(benchmark_id, experiment_name)}/{self.file_name}"
        )
        os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
        model.save_model(local_model_path)
        run[self.neptune_save_path].upload(local_model_path)

    @classmethod
    def load_from_neptune(cls, run: neptune.Run, **kwargs) -> xgb.Booster:
        if "benchmark_id" in kwargs:
            benchmark_id = kwargs["benchmark_id"]
        else:
            raise ValueError("benchmark_id is required")

        if "experiment_name" in kwargs:
            experiment_name = kwargs["experiment_name"]
        else:
            raise ValueError("experiment_name is required")
        local_model_path = (
            f"{cls.local_dir_path(benchmark_id, experiment_name)}/{cls.file_name}"
        )

        run[cls.neptune_save_path].download(local_model_path)
        model = xgb.Booster()
        model.load_model(local_model_path)
        return model


class TrainingConfigStorage(NeptuneStorage[TrainingConfig]):
    """
    Storage for the training config.
    """

    save_path_prefix: str = "config"

    def save_to_neptune(self, run: neptune.Run, **kwargs) -> None:
        if "config" in kwargs:
            config: TrainingConfig = kwargs["config"]
        else:
            raise ValueError("config is required")

        for field in TrainingConfig.model_fields:
            run[f"{self.save_path_prefix}/{field}"] = getattr(config, field)

    @classmethod
    def load_from_neptune(cls, run: neptune.Run, **kwargs) -> TrainingConfig:
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
