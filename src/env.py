from pydantic import BaseModel
from .models import AIModel, Backend
from pydantic_settings import BaseSettings, SettingsConfigDict


settings_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    case_sensitive=False,
    env_nested_delimiter="_",
    env_nested_max_split=1,
)


class AISettings(BaseModel):
    backend: Backend
    model_id: AIModel


class BackendBaseSettings(BaseModel):
    api_key: str
    base_url: str


class Settings(BaseSettings):
    ai: AISettings
    openai: BackendBaseSettings
    model_config = settings_config


settings = Settings()  # type: ignore

if __name__ == "__main__":
    print(settings.model_dump_json(indent=2))
