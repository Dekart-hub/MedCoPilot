from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from config import Settings


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Инициализирует чат-модель из настроек."""
    return ChatOpenAI(
        model=settings.openai.model,
        api_key=settings.openai.api_key,
        base_url=settings.openai.base_url,
        temperature=settings.openai.temperature,
        timeout=settings.openai.timeout,
        max_retries=settings.openai.max_retries,
    )
