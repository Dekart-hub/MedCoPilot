"""OpenAI-compatible LLM client backing the SOAP extractor.

Adapts an ``AsyncOpenAI`` client — pointed at the vLLM OpenAI server (T7) — to
the :class:`~soap.llm_client.LlmClient` port. vLLM constrains generation to a
JSON schema through ``response_format={"type": "json_schema", ...}``, so the
model returns a document the extractor can validate directly. This is the
composition point that wires settings into a ready-to-use extractor.
"""

from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from config.settings import Settings
from infra.vllm.deployment import MEDGEMMA_4B
from soap.llm_client import LlmClient
from soap.llm_extractor import LlmSoapExtractor
from soap.scorer import ConfidenceScorer, NullConfidenceScorer


class OpenAiLlmClient(LlmClient):
    """Schema-guided completions over an OpenAI-compatible vLLM endpoint."""

    def __init__(self, client: AsyncOpenAI, model: str, *, temperature: float = 0.0) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    async def complete_json(
        self, *, instructions: str, prompt: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(role="system", content=instructions),
            ChatCompletionUserMessageParam(role="user", content=prompt),
        ]
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "soap_extraction", "schema": schema},
            },
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM returned an empty response")
        parsed: Any = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("LLM response was not a JSON object")
        return parsed


def build_llm_client(settings: Settings) -> OpenAiLlmClient:
    """Construct an :class:`OpenAiLlmClient` from settings, with vLLM defaults."""
    client = AsyncOpenAI(
        base_url=settings.vllm_base_url or MEDGEMMA_4B.base_url(),
        # vLLM ignores the key unless started with ``--api-key``; a placeholder
        # keeps the OpenAI client, which requires a non-empty value, happy.
        api_key=settings.vllm_api_key or "not-needed",
    )
    return OpenAiLlmClient(client, settings.model_id or MEDGEMMA_4B.model_id)


def build_llm_extractor(
    settings: Settings, scorer: ConfidenceScorer | None = None
) -> LlmSoapExtractor:
    """Wire the LLM client and confidence scorer into a ready SOAP extractor."""
    return LlmSoapExtractor(build_llm_client(settings), scorer or NullConfidenceScorer())
