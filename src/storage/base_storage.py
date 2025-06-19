from typing import Generic, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound="BaseStorage")

ARTIFACTS_DIR = ".artifacts"


class BaseStorage(BaseModel, Generic[T]):
    def save_to_storage(self) -> None:
        raise NotImplementedError

    @classmethod
    def load_from_storage(cls: Type[T]) -> T:
        raise NotImplementedError
