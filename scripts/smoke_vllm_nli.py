"""Ручной smoke-тест VllmNliScorer против живого vLLM-эндпоинта.

Не входит в pytest: требует поднятого vLLM с моделью и доступа по сети.
Гоняет golden-набор src/soap/score/nli/golden_en.jsonl через calc_nli_score и
печатает score по каждой паре + грубую проверку ожидания (high/low).

Настройки через env:
    VLLM_BASE_URL    напр. http://localhost:8000
    VLLM_API_KEY     ключ (или пусто, если vllm serve без --api-key)
    NLI_MODEL_ID     напр. google/medgemma-4b-it
    NLI_TOKENIZER_ID обычно то же, что NLI_MODEL_ID

Запуск (внутри Colab/Kaggle рядом с поднятым vLLM):
    PYTHONPATH=src uv run --extra nli python scripts/smoke_vllm_nli.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from soap.score.nli import VllmNliScorer  # noqa: E402

GOLDEN = (
    Path(__file__).resolve().parent.parent
    / "src" / "soap" / "score" / "nli" / "golden_en.jsonl"
)

# Промпт откалиброван под MedGemma (instruct-модель gemma3): инструкция
# обёрнута в чат-шаблон Gemma (<start_of_turn>user … <start_of_turn>model).
# На голом completions-промпте (без шаблона) MedGemma «залипает» на yes и не
# отрабатывает отрицания — эмпирически 3/10 на golden_en.jsonl. С чат-шаблоном
# — 10/10, уверенное разделение (entailed→~1.0, contradicted→~0.0). Логит-паттерн
# при этом не меняется: читаем один токен yes/no сразу после метки model.
# Переопределяется через env NLI_PROMPT / NLI_PROMPT_SUFFIX (напр. под «родной»
# формат Qwen3-Reranker, у которого свой шаблон).
PROMPT = os.environ.get(
    "NLI_PROMPT",
    "<start_of_turn>user\n"
    "Read the Reference and the Claim. Does the Reference entail the Claim? "
    "If the Reference states or clearly implies the Claim, answer yes. "
    "If it contradicts or does not support the Claim, answer no. "
    "Answer with exactly one word, yes or no.\n\n",
)
PROMPT_SUFFIX = os.environ.get(
    "NLI_PROMPT_SUFFIX",
    "\n\n<end_of_turn>\n<start_of_turn>model\n",
)


def _load_pairs() -> list[dict]:
    with open(GOLDEN, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def main() -> None:
    base_url = os.environ["VLLM_BASE_URL"]
    model_id = os.environ["NLI_MODEL_ID"]
    scorer = VllmNliScorer(
        prompt=PROMPT,
        prompt_suffix=PROMPT_SUFFIX,
        model_id=model_id,
        tokenizer_id=os.environ.get("NLI_TOKENIZER_ID", model_id),
        vllm_base_url=base_url,
        api_key=os.environ.get("VLLM_API_KEY", ""),
    )
    print(f"vLLM: {base_url}  model: {model_id}\n")

    pairs = _load_pairs()
    scores = await asyncio.gather(
        *(scorer.calc_nli_score(p["inference"], p["reference"]) for p in pairs)
    )

    ok = 0
    for pair, score in zip(pairs, scores):
        expect = pair["expect"]
        # Грубая проверка: high -> ждём >0.5, low -> <0.5.
        hit = (score > 0.5) == (expect == "high")
        ok += hit
        mark = "OK " if hit else "!! "
        print(f"{mark}[{expect:4}] score={score:.3f}")
        print(f"     ref:   {pair['reference']}")
        print(f"     claim: {pair['inference']}\n")

    print(f"Совпало с ожиданием: {ok}/{len(pairs)}")


if __name__ == "__main__":
    asyncio.run(main())
