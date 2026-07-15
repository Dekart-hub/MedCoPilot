"""Unit tests for the OpenAI-compatible LLM adapter (no network)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from infra.llm import OpenAiLlmClient


class _RecordingCompletions:
    """Captures create() kwargs and returns a canned assistant message."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _client(content: str) -> tuple[OpenAiLlmClient, _RecordingCompletions]:
    completions = _RecordingCompletions(content)
    fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return OpenAiLlmClient(fake, "medgemma"), completions  # type: ignore[arg-type]


def _complete(client: OpenAiLlmClient, schema: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(client.complete_json(instructions="sys", prompt="hi", schema=schema))


def test_schema_is_enforced_via_response_format() -> None:
    # The live bug: vLLM silently ignores extra_body={"guided_json": ...}; the
    # schema must travel as a native response_format json_schema instead.
    schema = {"type": "object", "properties": {"problems": {"type": "array"}}}
    client, completions = _client('{"problems": []}')

    _complete(client, schema)

    response_format = completions.kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["schema"] is schema
    assert "extra_body" not in completions.kwargs


def test_empty_content_is_rejected() -> None:
    client, _ = _client("")
    client._client.chat.completions._content = None  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="empty"):
        _complete(client, {})


def test_non_object_json_is_rejected() -> None:
    client, _ = _client("[1, 2, 3]")

    with pytest.raises(ValueError, match="JSON object"):
        _complete(client, {})
