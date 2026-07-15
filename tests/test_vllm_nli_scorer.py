from __future__ import annotations

import asyncio
import math

import pytest

from nli import VllmNliScorer

# --- fakes: neither network nor a real model ------------------------------- #


class _FakeEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _FakeTokenizer:
    """Returns fixed ids for yes/no and decodes them back to strings."""

    _VOCAB = {"yes": 9820, "no": 2201}
    _INV = {9820: "yes", 2201: "no"}

    def encode(self, word: str, add_special_tokens: bool = True) -> _FakeEncoding:
        return _FakeEncoding([self._VOCAB[word]])

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
    """Mocks vLLM: returns preset top_logprobs and remembers the request."""

    def __init__(self, top_logprobs: dict[str, float]) -> None:
        self._payload = {"choices": [{"logprobs": {"top_logprobs": [top_logprobs]}}]}
        self.last_json: dict | None = None
        self.last_headers: dict | None = None

    async def post(self, url: str, json: dict, headers: dict) -> _FakeResponse:
        self.last_json = json
        self.last_headers = headers
        return _FakeResponse(self._payload)


def _scorer(client: _FakeClient, *, api_key: str = "sk-test") -> VllmNliScorer:
    return VllmNliScorer(
        prompt="judge: ",
        prompt_suffix="\nAnswer:",
        model_id="medgemma",
        tokenizer_id="unused-in-tests",
        vllm_base_url="http://vllm:8000",
        api_key=api_key,
        client=client,
        tokenizer=_FakeTokenizer(),
    )


def _run(coro):
    return asyncio.run(coro)


# --- tests ----------------------------------------------------------------- #


def test_score_is_softmax_of_yes_no_logprobs():
    # P(yes)=0.8, P(no)=0.2 in logprobs -> score should be exactly 0.8.
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
    # The model confidently said yes, 'no' did not make the top -> score -> 1.0.
    client = _FakeClient({"yes": math.log(0.95)})
    score = _run(_scorer(client).calc_nli_score("c", "r"))
    assert score == pytest.approx(1.0)


def test_softmax_is_stable_for_very_small_logprobs():
    client = _FakeClient({"yes": -10_000.0, "no": -10_001.0})
    score = _run(_scorer(client).calc_nli_score("c", "r"))
    assert score == pytest.approx(1 / (1 + math.exp(-1)))


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
    # The reference/inference pair really landed in the prompt.
    assert "reference text" in client.last_json["prompt"]
    assert "claim text" in client.last_json["prompt"]


def test_empty_api_key_omits_authorization_header():
    client = _FakeClient({"yes": math.log(0.6), "no": math.log(0.4)})
    _run(_scorer(client, api_key="").calc_nli_score("c", "r"))
    assert client.last_headers == {}


def test_answer_must_be_one_token():
    class SplitTokenizer(_FakeTokenizer):
        def encode(self, word: str, add_special_tokens: bool = True) -> _FakeEncoding:
            return _FakeEncoding([1, 2])

    with pytest.raises(ValueError, match="exactly one token"):
        VllmNliScorer(
            prompt="",
            prompt_suffix="",
            model_id="medgemma",
            tokenizer_id="unused-in-tests",
            vllm_base_url="http://vllm:8000",
            api_key="",
            client=_FakeClient({}),
            tokenizer=SplitTokenizer(),
        )
