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
from ehr.client import EhrClient
from icd.coder import IcdCoder
from infra.ehr import build_ehr_client
from infra.nli import build_nli_confidence_scorer
from infra.vllm.deployment import MEDGEMMA_4B
from soap.llm_client import LlmClient
from soap.llm_editor import SoapEditAgent
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


def build_soap_edit_agent(settings: Settings, ehr: EhrClient | None = None) -> SoapEditAgent:
    """Wire the LLM client and EHR into a SOAP edit agent (story #12).

    ``ehr`` is opt-in: an explicit client wins, otherwise the settings-driven
    mock (bundled fixture by default) is used. The ``model_id`` handed to the
    agent as generation metadata matches the one the LLM client actually calls.
    """
    return SoapEditAgent(
        build_llm_client(settings),
        ehr or build_ehr_client(settings),
        model_id=settings.model_id or MEDGEMMA_4B.model_id,
    )


def build_llm_extractor(
    settings: Settings,
    scorer: ConfidenceScorer | None = None,
    coder: IcdCoder | None = None,
) -> LlmSoapExtractor:
    """Wire the LLM client, confidence scorer and ICD coder into a SOAP extractor.

    ``coder`` is optional: left ``None``, assessment claims are extracted without
    ICD codings (backward-compatible default). ``scorer`` is opt-in: an explicit
    scorer wins; otherwise the NLI scorer is used only when
    ``settings.nli_confidence_enabled`` is set, and the default stays
    :class:`NullConfidenceScorer` (no tokenizer download).
    """
    return LlmSoapExtractor(
        build_llm_client(settings),
        scorer or _default_scorer(settings),
        coder=coder,
    )


def _default_scorer(settings: Settings) -> ConfidenceScorer:
    if settings.nli_confidence_enabled:
        return build_nli_confidence_scorer(settings)
    return NullConfidenceScorer()
