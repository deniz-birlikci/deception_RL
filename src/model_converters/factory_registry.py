from typing import Dict
from ..models import Backend, AIModel
from .base_model_converter_factory import BaseModelConverterFactory


class ModelConverterFactoryRegistry:

    _factory_classes: Dict[Backend, type[BaseModelConverterFactory]] = {}

    @classmethod
    def register_factory_class(
        cls, factory_class: type[BaseModelConverterFactory]
    ) -> None:
        cls._factory_classes[factory_class.backend] = factory_class

    @classmethod
    def create_factory(
        cls,
        backend: Backend,
        ai_model: AIModel | None = None,
    ) -> BaseModelConverterFactory:
        if backend not in cls._factory_classes:
            raise ValueError(f"No factory class registered for backend: {backend}")

        factory_class = cls._factory_classes[backend]
        return factory_class(ai_model=ai_model)

    @classmethod
    def get_supported_backends(cls) -> list[Backend]:
        return list(cls._factory_classes.keys())
