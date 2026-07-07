from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAISettings(BaseModel):
    """Настройки доступа к LLM (OpenAI-совместимый эндпоинт)."""

    api_key: str
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    temperature: float = 0.0


class ScoringSettings(BaseModel):
    """Quality-evaluation knobs (Tier 1 review flags)."""

    review_threshold: float = 0.6


class Settings(BaseSettings):
    """Корневые настройки приложения.

    Вложенные секции читаются из переменных окружения с разделителем ``__``,
    например ``OPENAI__API_KEY``, ``OPENAI__MODEL``.
    """

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai: OpenAISettings
    scoring: ScoringSettings = ScoringSettings()


@lru_cache
def get_settings() -> Settings:
    return Settings()
