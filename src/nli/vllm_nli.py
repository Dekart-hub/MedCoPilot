"""NLI scorer backed by MedGemma served in vLLM (a logits cross-encoder).

Low-level engine for FR-4 [#7/FR-4]: given two strings it returns
``P(entailment)`` in [0, 1]. It follows the Qwen3-Reranker pattern for vLLM --
instead of generating an answer, the model does a SINGLE forward pass and we
read the logprobs of the ``yes``/``no`` tokens at the first answer position and
take their softmax::

    score = exp(logprob_yes) / (exp(logprob_yes) + exp(logprob_no))

Reading one token from the logits (rather than parsing generated text) is fast
-- one position instead of autoregressive generation -- and deterministic.

The model lives in a separate vLLM service (OpenAI-compatible HTTP API), so no
torch is pulled into this package: locally we only need ``tokenizers`` to look
up the ``yes``/``no`` token ids. All inference happens on the vLLM side.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from tokenizers import Tokenizer

# Wraps the reference/inference pair between ``prompt`` and ``prompt_suffix``.
# Kept as a constant rather than baked into the prompt so the instruction
# template and the pair markup can vary independently.
_PAIR_TEMPLATE = "<Reference>: {reference}\n\n<Claim>: {inference}"


class VllmNliScorer:
    """Cross-encoder NLI scorer over a MedGemma vLLM endpoint.

    Contract (issue #13): the constructor takes the prompt template, the model
    and tokenizer ids, the vLLM address and key; the single working method is
    :meth:`calc_nli_score`, returning a float in [0, 1]. torch is not required
    -- the only heavy piece is the tokenizer, and even that is lazy and gets
    replaced by a fake in tests.

    ``client``/``tokenizer`` may be injected directly (fakes in tests), in which
    case neither ``httpx`` nor the Hugging Face Hub is touched.
    """

    def __init__(
        self,
        *,
        prompt: str,
        prompt_suffix: str,
        model_id: str,
        tokenizer_id: str,
        vllm_base_url: str,
        api_key: str,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self._prompt = prompt
        self._prompt_suffix = prompt_suffix
        self._model_id = model_id
        self._base_url = self._normalize_base_url(vllm_base_url)
        self._api_key = api_key
        self._timeout = timeout
        self._client = client

        if tokenizer is None:
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_pretrained(tokenizer_id)

        # Token ids of the answer plus their canonical string forms: the ids
        # restrict generation on the vLLM side (allowed_token_ids), the strings
        # locate the right logprobs in the response (top_logprobs is keyed by
        # strings).
        self._yes_id = self._first_token_id(tokenizer, "yes")
        self._no_id = self._first_token_id(tokenizer, "no")
        self._yes_str = tokenizer.decode([self._yes_id])
        self._no_str = tokenizer.decode([self._no_id])

    @staticmethod
    def _normalize_base_url(vllm_base_url: str) -> str:
        """Return the OpenAI-compatible ``/v1`` root, with or without a trailing ``/v1``.

        Clients on ``main`` build base URLs via ``VllmDeployment.base_url()``,
        which already ends in ``/v1``; a bare host is accepted too so the same
        engine works against either form.
        """
        base = vllm_base_url.rstrip("/")
        return base if base.endswith("/v1") else f"{base}/v1"

    @staticmethod
    def _first_token_id(tokenizer: Tokenizer, word: str) -> int:
        token_ids: list[int] = tokenizer.encode(word, add_special_tokens=False).ids
        if len(token_ids) != 1:
            raise ValueError(f"Answer {word!r} must map to exactly one token, got {token_ids}")
        return token_ids[0]

    async def calc_nli_score(self, inference: str, reference: str) -> float:
        """Probability that ``inference`` follows from ``reference`` (in [0, 1]).

        Async on purpose (unlike the bare ``-> float`` in the issue): the call is
        networked, and the surrounding scorer fans claims out via
        ``asyncio.gather`` so they run in parallel.
        """
        prompt = self._build_prompt(reference, inference)
        top_logprobs = await self._request_top_logprobs(prompt)
        return self._score_from_logprobs(top_logprobs)

    def _build_prompt(self, reference: str, inference: str) -> str:
        body = _PAIR_TEMPLATE.format(reference=reference, inference=inference)
        return f"{self._prompt}{body}{self._prompt_suffix}"

    async def _request_top_logprobs(self, prompt: str) -> dict[str, float]:
        """Call the vLLM completions API and pull the first position's logprobs.

        ``allowed_token_ids`` (a vLLM extension) forbids everything but
        ``yes``/``no``, so both tokens are guaranteed to appear in top_logprobs.
        """
        payload: dict[str, Any] = {
            "model": self._model_id,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 20,
            "allowed_token_ids": [self._yes_id, self._no_id],
        }
        # Empty api_key (local vLLM without --api-key) -> send no header at all:
        # httpx rejects "Bearer " with an empty value as invalid.
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        url = f"{self._base_url}/completions"

        if self._client is not None:
            resp = await self._client.post(url, json=payload, headers=headers)
        else:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return self._extract_top_logprobs(resp.json())

    @staticmethod
    def _extract_top_logprobs(data: dict[str, Any]) -> dict[str, float]:
        """Pull the first position's top_logprobs, failing loudly on any other shape.

        The completions response shape is taken on faith; if the real endpoint
        replies differently it matters to see the raw JSON, not a mute KeyError.
        """
        try:
            top_logprobs: dict[str, float] = data["choices"][0]["logprobs"]["top_logprobs"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "Unexpected vLLM response shape (no choices[0].logprobs."
                f"top_logprobs[0]): {str(data)[:500]}"
            ) from exc
        return top_logprobs

    def _score_from_logprobs(self, top_logprobs: dict[str, float]) -> float:
        yes_lp = self._lookup(top_logprobs, self._yes_str)
        no_lp = self._lookup(top_logprobs, self._no_str)
        if yes_lp is None and no_lp is None:
            raise ValueError("Neither 'yes' nor 'no' present in top_logprobs -- check model/prompt")
        if yes_lp is None:
            return 0.0
        if no_lp is None:
            return 1.0
        if math.isnan(yes_lp) or math.isnan(no_lp):
            raise ValueError("vLLM returned NaN for a yes/no log probability")
        if yes_lp == no_lp == -math.inf:
            raise ValueError("vLLM returned -inf for both yes and no")
        if yes_lp == -math.inf:
            return 0.0
        if no_lp == -math.inf:
            return 1.0
        if yes_lp >= no_lp:
            return 1.0 / (1.0 + math.exp(no_lp - yes_lp))
        ratio = math.exp(yes_lp - no_lp)
        return ratio / (1.0 + ratio)

    @staticmethod
    def _lookup(top_logprobs: dict[str, float], token: str) -> float | None:
        """Find a token's logprob, tolerant of leading spaces in its representation."""
        if token in top_logprobs:
            return top_logprobs[token]
        stripped = token.strip()
        for key, value in top_logprobs.items():
            if key.strip() == stripped:
                return value
        return None
