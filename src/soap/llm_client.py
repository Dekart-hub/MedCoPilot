"""Port for schema-guided LLM completions used by the SOAP extractor.

The extractor speaks in JSON-schema-constrained completions, not in any vendor's
API. A concrete adapter (an OpenAI-compatible vLLM client, T7) implements this
port at the infrastructure edge, keeping the extraction logic free of transport
and serialization concerns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LlmClient(ABC):
    """Returns a JSON object constrained to a caller-supplied JSON schema."""

    @abstractmethod
    async def complete_json(
        self, *, instructions: str, prompt: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Complete ``prompt`` under ``instructions``, returning a JSON object.

        The result must conform to ``schema``; enforcing that is the adapter's
        job (vLLM guided decoding). Raises on transport or protocol failure.
        """
        ...
