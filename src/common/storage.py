from typing import TypeVar, Type, Generic
from pydantic import BaseModel, Field
from common.config import TrainingConfig
from common.tracking import init_neptune_run
import neptune
import xgboost as xgb

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


class ModelStorage(NeptuneStorage[xgb.XGBClassifier]):
    """
    Storage for the artifact model.
    """

    neptune_save_path: str = "artifacts/model"
    file_name: str = "xgb_model.json"

    @staticmethod
    def local_path(benchmark_id: str, experiment_name: str) -> str:
        return f".artifacts/{benchmark_id}/{experiment_name}"

    def save_to_neptune(self, run: neptune.Run, **kwargs) -> None:
        if "benchmark_id" in kwargs:
            benchmark_id = kwargs["benchmark_id"]
        else:
            raise ValueError("benchmark_id is required")

        if "experiment_name" in kwargs:
            experiment_name = kwargs["experiment_name"]
        else:
            raise ValueError("experiment_name is required")
        local_path = (
            f"{self.local_path(benchmark_id, experiment_name)}/{self.file_name}"
        )

        run[self.neptune_save_path].upload(local_path)

    @classmethod
    def load_from_neptune(cls, run: neptune.Run, **kwargs) -> xgb.XGBClassifier:
        if "benchmark_id" in kwargs:
            benchmark_id = kwargs["benchmark_id"]
        else:
            raise ValueError("benchmark_id is required")

        if "experiment_name" in kwargs:
            experiment_name = kwargs["experiment_name"]
        else:
            raise ValueError("experiment_name is required")
        local_path = f"{cls.local_path(benchmark_id, experiment_name)}/{cls.file_name}"

        run[cls.neptune_save_path].download(local_path)
        model = xgb.XGBClassifier()
        model.load_model(local_path)
        return model


class TrainingConfigStorage(NeptuneStorage[TrainingConfig]):
    """
    Storage for the training config.
    """

    save_path: str = "config"

    def save_to_neptune(self, run: neptune.Run, **kwargs) -> None:
        for field in TrainingConfig.model_fields:
            run[f"{self.save_path}/{field}"] = getattr(TrainingConfig, field)

    @classmethod
    def load_from_neptune(cls, run: neptune.Run, **kwargs) -> TrainingConfig:
        try:
            config_dict = {}

            for field in TrainingConfig.model_fields:
                config_path = f"{cls.save_path}/{field}"
                try:
                    config_dict[field] = run[config_path].fetch()
                except Exception as e:
                    raise KeyError(
                        f"Missing expected config field in Neptune: {field}"
                    ) from e

            return TrainingConfig(**config_dict)

        except Exception as e:
            raise RuntimeError(f"Failed to load config from Neptune run {run._id}: {e}")
