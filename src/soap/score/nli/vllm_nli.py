"""NLI-скорер через MedGemma, поднятую в vLLM (cross-encoder на логитах).

Low-level движок для FR-4: принимает две строки и возвращает P(entailment)
в [0, 1]. Реализует паттерн Qwen3-Reranker для vLLM — вместо генерации текста
модель делает ОДИН проход, а мы читаем логиты токенов «yes»/«no» на первой
позиции ответа и берём softmax по ним:

    score = exp(logprob_yes) / (exp(logprob_yes) + exp(logprob_no))

Почему так, а не генерация ответа LLM:
- один токен вместо авторегрессионной генерации всего ответа -> быстро (важно
  для NFR-1: P99 <= 5 c на весь пайплайн);
- число берётся из логитов, а не из распарсенного текста -> детерминированно.

Модель живёт на сервере vLLM (OpenAI-совместимый HTTP API), поэтому в контейнер
НЕ тянется torch — только ``tokenizers`` локально, чтобы узнать id токенов
«yes»/«no». Инференс полностью на стороне vLLM.
"""

from __future__ import annotations

import math
from typing import Any

import httpx

# Обёртка вокруг reference/inference между ``prompt`` и ``prompt_suffix``.
# Вынесена в константу, а не «зашита» в prompt, чтобы шаблон промпта
# (инструкция) и разметку пары можно было менять независимо.
_PAIR_TEMPLATE = "<Reference>: {reference}\n\n<Claim>: {inference}"


class VllmNliScorer:
    """Cross-encoder NLI-скорер поверх vLLM-эндпоинта MedGemma.

    Контракт (issue #13): конструктор принимает шаблон промпта, id модели/
    токенизатора, адрес vLLM и ключ; единственный «рабочий» метод —
    :meth:`calc_nli_score`, возвращающий float в [0, 1]. torch не требуется:
    из тяжёлого — только токенизатор (и то ленивый, в тестах подменяется).

    ``client``/``tokenizer`` можно передать напрямую (для тестов — фейками),
    тогда ни ``httpx`` наружу, ни ``transformers`` не задействуются.
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
        tokenizer: Any | None = None,
    ) -> None:
        self._prompt = prompt
        self._prompt_suffix = prompt_suffix
        self._model_id = model_id
        self._base_url = vllm_base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._client = client

        if tokenizer is None:
            # Ленивый импорт: transformers нужен только для реального запуска,
            # torch он при этом не тянет (используем лишь токенизатор).
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

        # id токенов ответа и их канонические строковые формы: id — для
        # ограничения генерации на стороне vLLM (allowed_token_ids), строки —
        # для поиска нужных логитов в ответе (top_logprobs ключуется строками).
        self._yes_id = self._first_token_id(tokenizer, "yes")
        self._no_id = self._first_token_id(tokenizer, "no")
        self._yes_str = tokenizer.decode([self._yes_id])
        self._no_str = tokenizer.decode([self._no_id])

    @staticmethod
    def _first_token_id(tokenizer: Any, word: str) -> int:
        return tokenizer(word, add_special_tokens=False).input_ids[0]

    async def calc_nli_score(self, inference: str, reference: str) -> float:
        """Вероятность, что ``inference`` следует из ``reference`` (в [0, 1]).

        Асинхронный (в отличие от голого ``-> float`` в issue): вызов сетевой,
        а обёртка ``ConfidenceScorer`` гоняет claim'ы через ``asyncio.gather`` —
        так они реально идут параллельно. Синхронную сигнатуру это не нарушает
        по смыслу, но её стоит согласовать с автором таски.
        """
        prompt = self._build_prompt(reference, inference)
        top_logprobs = await self._request_top_logprobs(prompt)
        return self._score_from_logprobs(top_logprobs)

    def _build_prompt(self, reference: str, inference: str) -> str:
        body = _PAIR_TEMPLATE.format(reference=reference, inference=inference)
        return f"{self._prompt}{body}{self._prompt_suffix}"

    async def _request_top_logprobs(self, prompt: str) -> dict[str, float]:
        """Дёргает vLLM completions API и достаёт логпробы первой позиции.

        ``allowed_token_ids`` (расширение vLLM) запрещает всё, кроме «yes»/«no»,
        поэтому оба токена гарантированно оказываются в top_logprobs.
        """
        payload = {
            "model": self._model_id,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 20,
            "allowed_token_ids": [self._yes_id, self._no_id],
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        url = f"{self._base_url}/v1/completions"

        if self._client is not None:
            resp = await self._client.post(url, json=payload, headers=headers)
        else:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return self._extract_top_logprobs(data)

    @staticmethod
    def _extract_top_logprobs(data: dict) -> dict[str, float]:
        """Достаёт top_logprobs первой позиции, с понятной ошибкой при иной форме.

        Форму ответа vLLM мы приняли на веру (completions API) — если реальный
        эндпоинт отдаёт иначе, важно увидеть сырой JSON, а не немой KeyError.
        """
        try:
            return data["choices"][0]["logprobs"]["top_logprobs"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "Неожиданная форма ответа vLLM (нет choices[0].logprobs."
                f"top_logprobs[0]): {str(data)[:500]}"
            ) from exc

    def _score_from_logprobs(self, top_logprobs: dict[str, float]) -> float:
        yes_lp = self._lookup(top_logprobs, self._yes_str)
        no_lp = self._lookup(top_logprobs, self._no_str)
        if yes_lp is None and no_lp is None:
            raise ValueError(
                "Ни 'yes', ни 'no' нет в top_logprobs — проверь модель/промпт"
            )
        # Отсутствующий токен трактуем как -inf (exp -> 0): если модель уверенно
        # сказала yes, no могло не попасть в топ, и наоборот.
        yes_p = math.exp(yes_lp) if yes_lp is not None else 0.0
        no_p = math.exp(no_lp) if no_lp is not None else 0.0
        return yes_p / (yes_p + no_p)

    @staticmethod
    def _lookup(top_logprobs: dict[str, float], token: str) -> float | None:
        """Ищет логпроб токена, устойчиво к ведущим пробелам в представлении."""
        if token in top_logprobs:
            return top_logprobs[token]
        stripped = token.strip()
        for key, value in top_logprobs.items():
            if key.strip() == stripped:
                return value
        return None
