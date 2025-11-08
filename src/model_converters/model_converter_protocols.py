from typing import Protocol, TypeVar, runtime_checkable
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class DictConverter(Protocol[T]):

    def to_dict(self, data: T) -> dict: ...

    def from_dict(self, data: dict) -> T: ...


@runtime_checkable
class ListDictConverter(Protocol[T]):

    def to_list_dict(self, data: T) -> list[dict]: ...

    def from_dict(self, data: dict) -> T: ...
