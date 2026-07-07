from __future__ import annotations

from config.settings import get_settings


def _fresh_settings(monkeypatch, **env: str):
    monkeypatch.setenv("OPENAI__API_KEY", "sk-test")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    try:
        return get_settings()
    finally:
        get_settings.cache_clear()


def test_review_threshold_defaults_to_0_6(monkeypatch):
    settings = _fresh_settings(monkeypatch)
    assert settings.scoring.review_threshold == 0.6


def test_review_threshold_reads_env_override(monkeypatch):
    settings = _fresh_settings(monkeypatch, SCORING__REVIEW_THRESHOLD="0.8")
    assert settings.scoring.review_threshold == 0.8
