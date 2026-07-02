#!/usr/bin/env python3
"""Эвал кодирования диагнозов на golden set (английский, ICD-10-CM).

Две метрики, две стадии пайплайна:

  * recall@N — качество ретрива: попал ли правильный код в топ-N кандидатов.
    Ретрив отвечает за recall; если кода нет в пуле, реранкеру нечего выбирать.
  * top-1 accuracy — качество итогового выбора (``--llm``: полный нормализатор
    из DI, включая LLM-реранк; без флага — лексический top-1).

Каждый прогон сохраняется: полный отчёт (метрики + per-case сравнение) в
``runs/coding_eval/<timestamp>.json``, сводная строка дописывается в
``runs/coding_eval/history.jsonl`` — по ней видно динамику между
моделями/фиксами.

Запуск (PYTHONPATH=src уже экспортирует Makefile):
    uv run python scripts/eval_coding.py                  # только ретрив
    uv run python scripts/eval_coding.py --llm            # + LLM-этап (.env)
    uv run python scripts/eval_coding.py --show-misses    # разбор промахов
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, "src")

from config import CodingSettings  # noqa: E402
from shared.value_objects import Id  # noqa: E402
from soap.soap import SoapClaim, SoapEvidence, SoapNote  # noqa: E402

RECALL_AT = (1, 5, 10, 20, 30)


def load_golden(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_note(row: dict) -> SoapNote:
    def claim(text: str) -> SoapClaim:
        return SoapClaim(
            id=Id.new(),
            claim=text,
            evidence=SoapEvidence(text=text, turn_id=Id.new()),
        )

    return SoapNote(
        id=Id.new(),
        subjective=claim(row.get("subjective", "")),
        objective=claim(row.get("objective", "")),
        assessment=claim(row["assessment"]),
        plan=claim(""),
    )


def eval_retrieval(index, golden: list[dict], show_misses: bool) -> dict:
    max_n = max(RECALL_AT)
    hits = {n: 0 for n in RECALL_AT}
    misses: list[tuple[dict, list]] = []
    for row in golden:
        candidates = index.search(row["assessment"], top_n=max_n)
        codes = [c.code for c in candidates]
        rank = codes.index(row["code"]) + 1 if row["code"] in codes else None
        for n in RECALL_AT:
            if rank is not None and rank <= n:
                hits[n] += 1
        if rank is None or rank > 1:
            misses.append((row, candidates[:3]))

    total = len(golden)
    print(f"\n=== Ретрив: recall@N (golden: {total}) ===")
    for n in RECALL_AT:
        print(f"  recall@{n:<3} {hits[n]}/{total}  ({hits[n] / total:.0%})")

    if show_misses:
        print("\n--- Промахи top-1 ---")
        for row, top in misses:
            print(f"  {row['assessment']!r} (ждали {row['code']}):")
            for c in top:
                print(f"      {c.code:<9} {c.score:.2f}  {c.formulation}")

    return {f"recall@{n}": round(hits[n] / total, 3) for n in RECALL_AT}


async def eval_llm(
    golden: list[dict], concurrency: int, timeout: float
) -> tuple[dict, list[dict]]:
    from tqdm import tqdm

    from config import get_settings
    from di.container import build_normalizer
    from infra import build_chat_model
    from shared.prompts import InMemoryPromptStore
    from soap.coding import DEFAULT_CODING_PROMPTS

    settings = get_settings()
    normalizer = build_normalizer(
        settings.coding,
        model=build_chat_model(settings),
        prompts=InMemoryPromptStore(DEFAULT_CODING_PROMPTS),
    )

    semaphore = asyncio.Semaphore(concurrency)
    progress = tqdm(total=len(golden), desc="LLM-реранк", unit="кейс")

    async def one(row: dict):
        async with semaphore:
            try:
                coding = await asyncio.wait_for(
                    normalizer.normalize(to_note(row)), timeout=timeout
                )
            except TimeoutError:
                coding = None
            finally:
                progress.update(1)
            return row, coding

    results = await asyncio.gather(*(one(row) for row in golden))
    progress.close()

    correct = 0
    rows: list[dict] = []
    for row, coding in results:
        best = coding.best if coding else None
        ok = best is not None and best.code == row["code"]
        correct += ok
        mark = "+" if ok else "-"
        got = f"{best.code} {best.title!r}" if best else "— (таймаут)" if coding is None else "—"
        print(f"  [{mark}] {row['assessment']!r}: ждали {row['code']}, получили {got}")
        rows.append(
            {
                "assessment": row["assessment"],
                "expected": row["code"],
                "got": best.code if best else None,
                "got_title": best.title if best else None,
                "rationale": coding.rationale if coding else None,
                "ok": ok,
                "timeout": coding is None,
            }
        )
    total = len(golden)
    answered = sum(1 for r in rows if not r["timeout"])
    print(f"\n=== Итог: top-1 accuracy {correct}/{total} ({correct / total:.0%}) ===")
    summary = {
        "model": settings.openai.model,
        "base_url": settings.openai.base_url,
        "accuracy": round(correct / total, 3),
        "correct": correct,
        "answered": answered,
        "timeouts": total - answered,
    }
    return summary, rows


def save_report(report: dict, out_dir: str = "runs/coding_eval") -> str:
    """Полный отчёт — отдельным файлом, сводка — строкой в history.jsonl."""
    os.makedirs(out_dir, exist_ok=True)
    stamp = report["timestamp"].replace(":", "-").split(".")[0]
    path = os.path.join(out_dir, f"{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    summary = {k: v for k, v in report.items() if k != "results"}
    with open(os.path.join(out_dir, "history.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Эвал кодирования на golden set")
    parser.add_argument("--golden", default="data/golden/coding_en.jsonl")
    parser.add_argument("--llm", action="store_true", help="полный нормализатор из DI")
    parser.add_argument("--concurrency", type=int, default=4, help="параллельных LLM-вызовов")
    parser.add_argument("--timeout", type=float, default=150.0, help="сек на один кейс LLM")
    parser.add_argument("--show-misses", action="store_true")
    args = parser.parse_args()

    golden = load_golden(args.golden)

    from soap.coding import preprocess_en
    from soap.coding.coding import DEFAULT_ICD10CM_REF
    from soap.coding.retrieval import MkbIndex

    coding_settings = CodingSettings()
    index = MkbIndex.from_jsonl(
        f"{coding_settings.data_dir}/tabular.jsonl",
        f"{coding_settings.data_dir}/index.jsonl",
        base_ref=DEFAULT_ICD10CM_REF,
        tokenizer=preprocess_en.normalize,
    )
    report: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "golden": args.golden,
        "cases": len(golden),
        "classifier_version": index.classifier.version,
        "retrieval_top_n": coding_settings.retrieval_top_n,
        "retrieval": eval_retrieval(index, golden, args.show_misses),
    }

    if args.llm:
        llm_summary, rows = asyncio.run(eval_llm(golden, args.concurrency, args.timeout))
        report["llm"] = llm_summary
        report["results"] = rows

    path = save_report(report)
    print(f"\nОтчёт: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
