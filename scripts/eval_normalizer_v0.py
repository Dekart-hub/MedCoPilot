#!/usr/bin/env python3
"""Бутстрап-оценка recall@k для лексического нормализатора (v0).

Строит индекс из выгруженных справочников НСИ и проверяет, находит ли ретрив
правильный код, если формулировку Тома 3 слегка «испортить» (выкинуть слово,
перемешать порядок) — грубый прокси разговорного перефраза.

ВАЖНО: это нижняя граница, а не настоящая метрика. Перестановка/выкид слова —
не то же самое, что живой текст врача. Честный замер требует размеченных пар
«ассессмент -> код» (вручную или LLM-перефразом). Скрипт даёт первую цифру,
от которой плясать при решении «нужны ли эмбеддинги».

Запуск (после `fetch_mkb_nsi.py --all --out data`):
    python3 scripts/eval_normalizer_v0.py --vol1 data/mkb10_vol1.jsonl \
        --vol3 data/mkb10_vol3_index.jsonl --sample 2000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from soap.coding.retrieval import MkbIndex  # noqa: E402

_TOP_KS = (1, 3, 5, 10)


def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _perturb(name: str, rng: random.Random) -> str:
    """Грубый прокси перефраза: выкинуть слово и/или перемешать порядок."""
    words = name.split()
    if len(words) > 2 and rng.random() < 0.7:
        words.pop(rng.randrange(len(words)))
    rng.shuffle(words)
    return " ".join(words)


def main() -> int:
    parser = argparse.ArgumentParser(description="recall@k для нормализатора v0")
    parser.add_argument("--vol1", default="data/mkb10_vol1.jsonl")
    parser.add_argument("--vol3", default="data/mkb10_vol3_index.jsonl")
    parser.add_argument("--sample", type=int, default=2000,
                        help="сколько формулировок взять под оценку")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for path in (args.vol1, args.vol3):
        if not os.path.exists(path):
            print(f"Нет файла {path}. Сначала: fetch_mkb_nsi.py --all", file=sys.stderr)
            return 2

    t0 = time.time()
    vol3 = _read_jsonl(args.vol3)
    index = MkbIndex.from_records(_read_jsonl(args.vol1), vol3)
    print(f"Индекс построен за {time.time() - t0:.1f}с, формулировок: {len(vol3)}",
          file=sys.stderr)

    rng = random.Random(args.seed)
    pairs = [(r["S_NAME"], r["ICD-10"]) for r in vol3 if r.get("S_NAME") and r.get("ICD-10")]
    sample = rng.sample(pairs, min(args.sample, len(pairs)))

    hits = {k: 0 for k in _TOP_KS}
    max_k = max(_TOP_KS)
    for name, gold in sample:
        query = _perturb(name, rng)
        codes = [c.code for c in index.search(query, top_n=max_k)]
        for k in _TOP_KS:
            if gold in codes[:k]:
                hits[k] += 1

    n = len(sample)
    print(f"\nrecall@k на {n} формулировках (perturbed):")
    for k in _TOP_KS:
        print(f"  recall@{k:<2} = {hits[k] / n:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
