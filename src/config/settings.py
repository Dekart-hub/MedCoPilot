from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Каждая секция — самостоятельный BaseSettings со своим env_prefix: имена
# переменных получаются плоскими и стандартными (OPENAI_API_KEY — ровно то,
# что читает OpenAI SDK), без вложенного разделителя "__".
_ENV = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


class OpenAISettings(BaseSettings):
    """Настройки доступа к LLM (OpenAI-совместимый эндпоинт).

    Переменные: ``OPENAI_API_KEY``, ``OPENAI_MODEL``, ``OPENAI_BASE_URL``,
    ``OPENAI_TEMPERATURE``, ``OPENAI_TIMEOUT``, ``OPENAI_MAX_RETRIES``.
    Для OpenRouter достаточно подменить
    ``OPENAI_BASE_URL=https://openrouter.ai/api/v1`` и ключ.

    ``timeout``/``max_retries`` обязательны для агрегаторов вроде OpenRouter:
    отдельные запросы там иногда виснут, и без таймаута весь пайплайн ждёт их
    вечно вместо быстрого ретрая.
    """

    model_config = SettingsConfigDict(env_prefix="OPENAI_", **_ENV)

    api_key: str
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    temperature: float = 0.0
    timeout: float = 45.0
    max_retries: int = 2


class CodingSettings(BaseSettings):
    """Нормализатор диагнозов в код классификатора.

    ``language`` выбирает справочник и препроцессинг: ``en`` -> ICD-10-CM
    (``tabular.jsonl``/``index.jsonl`` в ``data_dir``), ``ru`` -> МКБ-10 НСИ
    (``mkb10_vol1.jsonl``/``mkb10_vol3_index.jsonl``). Если файлов нет —
    нормализатор деградирует до заглушки, приложение всё равно поднимается.

    ``retrieval_top_n`` — размер пула кандидатов лексического ретрива (пул
    работает на recall: на golden set recall@20 = 100%). ``llm_rerank``
    включает Tier 2 — LLM выбирает итоговый код из пула; при выключенном
    флаге остаёмся на лексическом top-1 (оффлайн, без затрат на LLM).

    Переменные: ``CODING_LANGUAGE``, ``CODING_DATA_DIR``,
    ``CODING_RETRIEVAL_TOP_N``, ``CODING_LLM_RERANK``.
    """

    model_config = SettingsConfigDict(env_prefix="CODING_", **_ENV)

    language: Literal["en", "ru"] = "en"
    data_dir: str = "data/icd10cm"
    retrieval_top_n: int = 20
    llm_rerank: bool = True


class Settings(BaseSettings):
    """Корневые настройки приложения.

    Секции собираются каждая из своих плоских переменных окружения
    (``OPENAI_*``, ``CODING_*``) — см. докстринги секций.
    """

    model_config = SettingsConfigDict(**_ENV)

    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    coding: CodingSettings = Field(default_factory=CodingSettings)


@lru_cache
def get_settings() -> Settings:
    return Settings()
