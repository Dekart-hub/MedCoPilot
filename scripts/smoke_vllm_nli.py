"""Manual smoke test of VllmNliScorer against a live vLLM endpoint.

Not part of pytest: it needs a running vLLM with the model and network access.
Runs the golden set src/nli/golden_en.jsonl through calc_nli_score and prints a
score per pair plus a coarse expectation check (high/low).

Configured via env:
    VLLM_BASE_URL     e.g. http://localhost:8001/v1
    VLLM_API_KEY      key (or empty, if vllm serve without --api-key)
    MODEL_ID          e.g. google/medgemma-4b-it (defaults to MEDGEMMA_4B)
    NLI_TOKENIZER_ID  usually the same as MODEL_ID

Run (next to a running vLLM):
    VLLM_BASE_URL=http://localhost:8001/v1 MODEL_ID=google/medgemma-4b-it \
        PYTHONPATH=src uv run python scripts/smoke_vllm_nli.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from infra.vllm.deployment import MEDGEMMA_4B_MODEL_ID  # noqa: E402
from nli import VllmNliScorer  # noqa: E402

GOLDEN = Path(__file__).resolve().parent.parent / "src" / "nli" / "golden_en.jsonl"

# Prompt calibrated for MedGemma (the instruct gemma3 model): the instruction is
# wrapped in the Gemma chat template (<start_of_turn>user ... <start_of_turn>model).
# On a bare completions prompt (no template) MedGemma sticks to "yes" and ignores
# negations -- empirically 3/10 on golden_en.jsonl. With the chat template it is
# 10/10, clean separation (entailed -> ~1.0, contradicted -> ~0.0). The logits
# pattern is unchanged: we read one yes/no token right after the model tag.
# Overridable via env NLI_PROMPT / NLI_PROMPT_SUFFIX (e.g. for Qwen3-Reranker's
# native template).
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
    model_id = os.environ.get("MODEL_ID") or os.environ.get("NLI_MODEL_ID", MEDGEMMA_4B_MODEL_ID)
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
    for pair, score in zip(pairs, scores, strict=True):
        expect = pair["expect"]
        # Coarse check: high -> expect >0.5, low -> <0.5.
        hit = (score > 0.5) == (expect == "high")
        ok += hit
        mark = "OK " if hit else "!! "
        print(f"{mark}[{expect:4}] score={score:.3f}")
        print(f"     ref:   {pair['reference']}")
        print(f"     claim: {pair['inference']}\n")

    print(f"Matched expectation: {ok}/{len(pairs)}")


if __name__ == "__main__":
    asyncio.run(main())
