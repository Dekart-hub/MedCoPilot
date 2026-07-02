from __future__ import annotations

import pytest

from config import CodingSettings, OpenAISettings, Settings


def test_openai_reads_standard_flat_env_names(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_TEMPERATURE", "0.3")

    s = OpenAISettings(_env_file=None)
    assert s.api_key == "sk-test"
    assert s.model == "openai/gpt-4o-mini"
    assert s.base_url == "https://openrouter.ai/api/v1"
    assert s.temperature == 0.3


def test_coding_reads_standard_flat_env_names(monkeypatch):
    monkeypatch.setenv("CODING_LANGUAGE", "ru")
    monkeypatch.setenv("CODING_RETRIEVAL_TOP_N", "30")
    monkeypatch.setenv("CODING_LLM_RERANK", "false")

    c = CodingSettings(_env_file=None)
    assert c.language == "ru"
    assert c.retrieval_top_n == 30
    assert c.llm_rerank is False


def test_settings_assembles_sections_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    s = Settings(_env_file=None)
    assert s.openai.api_key == "sk-test"
    assert s.coding.language == "en"  # дефолты секции живы


def test_missing_api_key_fails_loudly(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(Exception, match="api_key"):
        OpenAISettings(_env_file=None)
