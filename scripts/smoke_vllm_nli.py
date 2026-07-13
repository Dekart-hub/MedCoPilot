"""Ручной smoke-тест VllmNliScorer против живого vLLM-эндпоинта.

Не входит в pytest: требует поднятого vLLM с моделью и доступа по сети.
Гоняет golden-набор data/golden/nli_pairs_en.jsonl через calc_nli_score и
печатает score по каждой паре + грубую проверку ожидания (high/low).

Настройки через env:
    VLLM_BASE_URL    напр. http://localhost:8000
    VLLM_API_KEY     ключ (или пусто, если vllm serve без --api-key)
    NLI_MODEL_ID     напр. google/medgemma-4b-it
    NLI_TOKENIZER_ID обычно то же, что NLI_MODEL_ID

Запуск (внутри Colab/Kaggle рядом с поднятым vLLM):
    PYTHONPATH=src python scripts/smoke_vllm_nli.py
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

# Черновой промпт под logits-паттерн: обрывается ровно там, где модель
# должна выдать один токен yes/no. Подбирается эмпирически под модель;
# можно переопределить через env NLI_PROMPT / NLI_PROMPT_SUFFIX (например,
# под «родной» формат Qwen3-Reranker).
PROMPT = os.environ.get(
    "NLI_PROMPT",
    "You are a clinical NLI judge. Decide whether the Claim logically follows "
    "from the Reference. Answer only yes or no.\n\n",
)
PROMPT_SUFFIX = os.environ.get("NLI_PROMPT_SUFFIX", "\n\nAnswer (yes or no):")


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
