from typing import Dict, Any
from ..models import Backend, AIModel
from .base_agent import BaseAgent


class AgentRegistry:

    _agent_classes: Dict[Backend, type[BaseAgent]] = {}

    @classmethod
    def register_agent_class(cls, agent_class: type[BaseAgent]) -> None:
        cls._agent_classes[agent_class.backend] = agent_class

    @classmethod
    def create_agent(
        cls,
        backend: Backend,
        ai_model: AIModel | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> BaseAgent:
        if backend not in cls._agent_classes:
            raise ValueError(f"No agent class registered for backend: {backend}")

        agent_class = cls._agent_classes[backend]
        return agent_class(
            ai_model=ai_model,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    @classmethod
    def get_supported_backends(cls) -> list[Backend]:
        return list(cls._agent_classes.keys())
