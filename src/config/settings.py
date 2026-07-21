"""Application settings loaded from the environment and an optional ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the service.

    Values are read from environment variables (matched case-insensitively) or a
    local ``.env`` file. The placeholder fields are wired now so later tasks can
    populate them without reshaping the config surface.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "dev"

    database_url: str | None = None
    vllm_base_url: str | None = None
    vllm_api_key: str | None = None
    model_id: str | None = None

    # NLI groundedness confidence scorer (T12). Off by default so the plain
    # extractor never triggers a tokenizer download; opt in with the flag or by
    # passing an explicit scorer. Prompt/tokenizer overrides fall back to the
    # MedGemma defaults wired in ``infra.nli``.
    nli_confidence_enabled: bool = False
    nli_tokenizer_id: str | None = None
    nli_prompt: str | None = None
    nli_prompt_suffix: str | None = None

    # Path to the mock EHR's ``patient_id -> context`` JSON. Unset ⇒ the bundled
    # fixture wired by ``infra.ehr.build_ehr_client``.
    ehr_mock_path: Path | None = None

    fhir_base_url: str = "http://localhost:8080/fhir"
    fhir_identifier_system: str = "urn:medcopilot:ehr-publication"
    fhir_timeout_seconds: float = 10.0
    fhir_dispatcher_enabled: bool = False
    fhir_dispatcher_poll_seconds: float = 1.0
    fhir_dispatcher_batch_size: int = 10
    fhir_retry_initial_seconds: float = 1.0
    fhir_retry_max_seconds: float = 300.0


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
