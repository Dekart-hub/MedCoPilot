from __future__ import annotations

import asyncio
import math

import pytest

from soap.score.nli import VllmNliScorer


# --- фейки: ни сети, ни реальной модели ----------------------------------- #


class _FakeTokenizerOut:
    def __init__(self, ids: list[int]) -> None:
        self.input_ids = ids


class _FakeTokenizer:
    """Отдаёт фиксированные id для yes/no и обратно декодирует их в строки."""

    _VOCAB = {"yes": 9820, "no": 2201}
    _INV = {9820: "yes", 2201: "no"}

    def __call__(self, word: str, add_special_tokens: bool = True) -> _FakeTokenizerOut:
        return _FakeTokenizerOut([self._VOCAB[word]])

    def decode(self, ids: list[int]) -> str:
        return self._INV[ids[0]]


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """Мокает vLLM: возвращает заранее заданные top_logprobs, запоминает запрос."""

    def __init__(self, top_logprobs: dict[str, float]) -> None:
        self._payload = {
            "choices": [{"logprobs": {"top_logprobs": [top_logprobs]}}]
        }
        self.last_json: dict | None = None

    async def post(self, url: str, json: dict, headers: dict) -> _FakeResponse:
        self.last_json = json
        return _FakeResponse(self._payload)


def _scorer(client: _FakeClient) -> VllmNliScorer:
    return VllmNliScorer(
        prompt="judge: ",
        prompt_suffix="\nAnswer:",
        model_id="medgemma",
        tokenizer_id="unused-in-tests",
        vllm_base_url="http://vllm:8000",
        api_key="sk-test",
        client=client,
        tokenizer=_FakeTokenizer(),
    )


def _run(coro):
    return asyncio.run(coro)


# --- собственно тесты ------------------------------------------------------ #


def test_score_is_softmax_of_yes_no_logprobs():
    # P(yes)=0.8, P(no)=0.2 в логпробах -> score должен быть ровно 0.8.
    client = _FakeClient({"yes": math.log(0.8), "no": math.log(0.2)})
    score = _run(_scorer(client).calc_nli_score("claim", "reference"))
    assert score == pytest.approx(0.8)


def test_score_always_in_unit_range():
    for yes_p, no_p in [(0.99, 0.01), (0.5, 0.5), (0.01, 0.99), (0.3, 0.7)]:
        client = _FakeClient({"yes": math.log(yes_p), "no": math.log(no_p)})
        score = _run(_scorer(client).calc_nli_score("c", "r"))
        assert 0.0 <= score <= 1.0


def test_contradiction_scores_below_half():
    client = _FakeClient({"yes": math.log(0.1), "no": math.log(0.9)})
    score = _run(_scorer(client).calc_nli_score("c", "r"))
    assert score < 0.5


def test_missing_no_token_treated_as_zero_probability():
    # Модель уверенно сказала yes, 'no' не попал в топ -> score -> 1.0.
    client = _FakeClient({"yes": math.log(0.95)})
    score = _run(_scorer(client).calc_nli_score("c", "r"))
    assert score == pytest.approx(1.0)


def test_both_tokens_missing_raises():
    client = _FakeClient({"maybe": math.log(0.5)})
    with pytest.raises(ValueError):
        _run(_scorer(client).calc_nli_score("c", "r"))


def test_request_restricts_generation_to_yes_no_ids():
    client = _FakeClient({"yes": math.log(0.6), "no": math.log(0.4)})
    _run(_scorer(client).calc_nli_score("claim text", "reference text"))

    assert client.last_json is not None
    assert client.last_json["max_tokens"] == 1
    assert client.last_json["allowed_token_ids"] == [9820, 2201]
    # Пара reference/inference реально попала в промпт.
    assert "reference text" in client.last_json["prompt"]
    assert "claim text" in client.last_json["prompt"]
