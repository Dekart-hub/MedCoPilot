#!/usr/bin/env python3
"""Смоук-прогон полного пайплайна на реальном корпусе CodiEsp (CLEF eHealth 2020).

CodiEsp — испанские клинические случаи с кодами МКБ-10, проставленными
профессиональными кодировщиками; используем официальный английский перевод
(``text_files_en``) и gold-коды диагнозов (``*D.tsv``). Источник:
https://zenodo.org/records/3837305 (CC-BY 4.0).

Метрики — официальные для CodiEsp-D: micro-precision/recall/F1 по парам
«документ-код». Ориентиры на английском переводе: PLM-ICD (supervised)
micro-F1 ~0.216, GPT-4 tree-search zero-shot 0.157 (Boyle et al., 2023).
Считаем два уровня: точное совпадение кода и 3-значная рубрика (gold — это
CIE-10-ES, испанская модификация МКБ-10; с ICD-10-CM почти всюду совпадает,
но на полной глубине кода возможны расхождения не по нашей вине).

Пайплайн под многокодовый бенчмарк: экстракция SOAP-нот как в продукте, но
кодировщик выбирает ВСЕ обоснованные текстом коды из пула ретрива (в продукте
реранкер выбирает один ведущий диагноз — здесь это дало бы заниженный recall
by design). Предсказание документа — объединение кодов по всем нотам.

Запуск:
    uv run python scripts/eval_codiesp.py --data <корень codiesp> --n 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, "src")

from datetime import datetime as _dt  # noqa: E402

from dialogue import Dialogue, DialogueTurn  # noqa: E402
from shared.value_objects import Id  # noqa: E402


def load_cases(root: str, split: str, n: int) -> list[dict]:
    """Первые n кейсов сплита: id, английский текст, gold-коды диагнозов."""
    gold_by_case: dict[str, list[str]] = {}
    with open(os.path.join(root, split, f"{split}D.tsv"), encoding="utf-8") as f:
        for line in f:
            case_id, _, code = line.strip().partition("\t")
            if code:
                gold_by_case.setdefault(case_id, []).append(code.upper())

    cases = []
    text_dir = os.path.join(root, split, "text_files_en")
    for case_id in sorted(gold_by_case)[:n]:
        path = os.path.join(text_dir, f"{case_id}.txt")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            cases.append({"id": case_id, "text": f.read(), "gold": gold_by_case[case_id]})
    return cases


def to_dialogue(text: str) -> Dialogue:
    """Клинический случай как «диалог» из реплик врача: одна строка — реплика.

    ``Dialogue.from_text`` здесь не подходит: он считает первое слово строки
    меткой роли и откусил бы его от повествовательного текста.
    """
    now = _dt.now(timezone.utc)
    turns = [
        DialogueTurn(id=Id.new(), role="medic", content=line.strip(), timestamp=now)
        for line in text.splitlines()
        if line.strip()
    ]
    return Dialogue(id=Id.new(), turns=turns, created_at=now)


def category(code: str) -> str:
    return code.split(".")[0]


MULTI_CODE_PROMPT_KEY = "coding.multi"

# Бенчмарк-режим: выбрать все обоснованные коды, а не один ведущий диагноз.
MULTI_CODE_PROMPT = (
    "Ты — медицинский кодировщик. Ниже SOAP-нота из истории болезни и "
    "кандидаты кодов ICD-10-CM, найденные лексическим поиском.\n\n"
    "Subjective: {{ subjective }}\n"
    "Objective: {{ objective }}\n"
    "Assessment: {{ assessment }}\n"
    "Plan: {{ plan }}\n\n"
    "Кандидаты:\n{{ candidates }}\n\n"
    "Выбери ВСЕ коды, обоснованные текстом ноты: основной диагноз, "
    "сопутствующие болезни, значимые симптомы, анамнез. Только коды из "
    "списка выше. Если ничего не подходит — верни пустой список."
)


async def run(cases: list[dict], concurrency: int, timeout: float) -> list[dict]:
    from pydantic import BaseModel
    from tqdm import tqdm

    from config import get_settings
    from infra import build_chat_model
    from shared.langgraph import LangGraphAgent
    from shared.prompts import InMemoryPromptStore
    from soap.coding import preprocess_en
    from soap.coding.coding import DEFAULT_ICD10CM_REF
    from soap.coding.retrieval import MkbIndex
    from soap.extractor import DEFAULT_PROMPTS, LlmSoapExtractor, build_graph

    class MultiCodeOut(BaseModel):
        codes: list[str]

    settings = get_settings()
    model = build_chat_model(settings)
    prompts = InMemoryPromptStore(
        {**DEFAULT_PROMPTS, MULTI_CODE_PROMPT_KEY: MULTI_CODE_PROMPT}
    )
    extractor = LlmSoapExtractor(LangGraphAgent(build_graph(model, prompts)))
    coder = model.with_structured_output(MultiCodeOut)
    index = MkbIndex.from_jsonl(
        f"{settings.coding.data_dir}/tabular.jsonl",
        f"{settings.coding.data_dir}/index.jsonl",
        base_ref=DEFAULT_ICD10CM_REF,
        tokenizer=preprocess_en.normalize,
    )

    async def code_note(note) -> set[str]:
        """Пул кандидатов по всем секциям ноты -> LLM выбирает все обоснованные."""
        pool: dict[str, str] = {}
        for text in (note.assessment.claim, note.subjective.claim, note.objective.claim):
            if text:
                for c in index.search(text, top_n=15):
                    pool.setdefault(c.code, index.name_of(c.code) or c.formulation)
        if not pool:
            return set()
        allowed = set(pool)
        for code in list(pool):
            allowed.update(index.parent_chain(code))
            allowed.update(ch.code for ch in index.children_of(code))
        rendered = "\n".join(f"- {code}: {title}" for code, title in sorted(pool.items()))
        prompt = await prompts.get(
            MULTI_CODE_PROMPT_KEY,
            subjective=note.subjective.claim or "—",
            objective=note.objective.claim or "—",
            assessment=note.assessment.claim or "—",
            plan=note.plan.claim or "—",
            candidates=rendered,
        )
        result = await coder.ainvoke(prompt)
        return {c.upper() for c in result.codes if c.upper() in allowed}

    semaphore = asyncio.Semaphore(concurrency)
    progress = tqdm(total=len(cases), desc="CodiEsp", unit="кейс")

    async def one(case: dict) -> dict:
        async with semaphore:
            try:
                async with asyncio.timeout(timeout):
                    report = await extractor.extract(to_dialogue(case["text"]))
                    note_codes = await asyncio.gather(
                        *(code_note(note) for note in report.soap_notes)
                    )
                predicted = sorted(set().union(*note_codes)) if note_codes else []
                assessments = [n.assessment.claim for n in report.soap_notes]
                error = None
            except Exception as e:  # таймаут кейса или сбой LLM — фиксируем, не падаем
                predicted, assessments, error = [], [], f"{type(e).__name__}: {e}"
            finally:
                progress.update(1)

            gold = set(case["gold"])
            return {
                "id": case["id"],
                "gold": sorted(gold),
                "predicted": predicted,
                "assessments": assessments,
                "exact_hits": sorted(set(predicted) & gold),
                "category_hits": sorted(
                    {category(c) for c in predicted} & {category(g) for g in gold}
                ),
                "error": error,
            }

    results = await asyncio.gather(*(one(case) for case in cases))
    progress.close()
    return list(results)


def micro_metrics(results: list[dict], to_key) -> dict:
    """Micro-P/R/F1 по парам «документ-код» (официальная метрика CodiEsp-D)."""
    tp = fp = fn = 0
    for r in results:
        pred = {to_key(c) for c in r["predicted"]}
        gold = {to_key(c) for c in r["gold"]}
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp, "fp": fp, "fn": fn,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Смоук-прогон на CodiEsp")
    parser.add_argument("--data", required=True, help="корень final_dataset_v4_to_publish")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--n", type=int, default=10, help="сколько кейсов взять")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=420.0, help="сек на кейс целиком")
    args = parser.parse_args()

    cases = load_cases(args.data, args.split, args.n)
    print(f"Кейсов: {len(cases)} ({args.split})", file=sys.stderr)

    results = asyncio.run(run(cases, args.concurrency, args.timeout))

    failed = sum(1 for r in results if r["error"])
    for r in results:
        status = "ошибка: " + r["error"] if r["error"] else (
            f"предсказано {len(r['predicted'])}, gold {len(r['gold'])}, "
            f"точных {len(r['exact_hits'])}, рубрик {len(r['category_hits'])}"
        )
        print(f"\n[{r['id']}] {status}")
        print(f"  gold:        {', '.join(r['gold'])}")
        print(f"  предсказано: {', '.join(r['predicted']) or '—'}")
        if r["exact_hits"]:
            print(f"  попадания:   {', '.join(r['exact_hits'])}")

    exact = micro_metrics(results, to_key=lambda c: c)
    by_cat = micro_metrics(results, to_key=category)
    print(f"\n=== Micro-метрики (документ-код), кейсов: {len(results)}, сбоев: {failed} ===")
    print(f"  точный код: P={exact['precision']} R={exact['recall']} F1={exact['f1']}")
    print(f"  3-зн. рубрика: P={by_cat['precision']} R={by_cat['recall']} F1={by_cat['f1']}")
    print("  ориентиры (en, точный код): GPT-4 tree-search F1=0.157, PLM-ICD F1≈0.216")

    from config import get_settings

    out_dir = "runs/coding_eval"
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat().replace(":", "-").split(".")[0]
    path = os.path.join(out_dir, f"codiesp_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "dataset": "CodiEsp v4 (en), zenodo 3837305",
                "split": args.split,
                "cases": len(results),
                "model": get_settings().openai.model,
                "failed": failed,
                "micro_exact": exact,
                "micro_category": by_cat,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Отчёт: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
