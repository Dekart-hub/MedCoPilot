#!/usr/bin/env python3
"""Парсер ICD-10-CM (NCHS/CMS) XML -> JSONL для лексического индекса.

Английский аналог выгрузки НСИ: приводит официальные файлы FY к той же плоской
схеме, что ждёт ``MkbIndex.from_records``, чтобы переиспользовать BM25-пайплайн
без изменений.

  * Tabular List  (icd10cm-tabular-YYYY.xml) -> «Том 1»: MKB_CODE / MKB_NAME /
    ID / ID_PARENT / ACTUAL. Иерархия берётся из вложенности <diag>.
  * Alphabetic Index (icd10cm-index-YYYY.xml) -> «Том 3»: S_NAME / ICD-10.
    S_NAME — это полный путь термов (предки + лист), чтобы в мешок слов попали
    слова-родители. Перекрёстные ссылки <see>/<seeAlso> без кода пропускаем.

Рядом с каждым JSONL пишется .meta.json с версией (фискальный год) — источник
provenance для ClassifierRef.

Запуск:
    python scripts/parse_icd10cm.py \
        --tabular data/icd10cm/icd10cm-tabular-2026.xml \
        --index data/icd10cm/icd10cm-index-2026.xml \
        --out data/icd10cm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET


def _text(elem: ET.Element | None) -> str:
    """Весь текст элемента, включая <nemod> и прочие вложения, одной строкой."""
    if elem is None:
        return ""
    return " ".join("".join(elem.itertext()).split())


# --- Tabular List («Том 1») ------------------------------------------------ #


def parse_tabular(path: str) -> tuple[list[dict], str]:
    root = ET.parse(path).getroot()
    version = _text(root.find("version")) or "unknown"
    rows: list[dict] = []

    def walk(parent: ET.Element, parent_code: str | None) -> None:
        for diag in parent.findall("diag"):
            code = _text(diag.find("name"))
            if not code:
                continue
            rows.append(
                {
                    "ID": code,  # код самодостаточен как идентификатор записи
                    "MKB_CODE": code,
                    "MKB_NAME": _text(diag.find("desc")),
                    "ID_PARENT": parent_code,  # None у рубрик верхнего уровня
                    "ACTUAL": "1",  # в файле только актуальные коды FY
                }
            )
            walk(diag, code)  # вложенные <diag> -> дети по иерархии

    # chapter -> section -> diag (далее рекурсия по вложенным diag)
    for chapter in root.findall("chapter"):
        for section in chapter.findall("section"):
            walk(section, None)
    return rows, version


# --- Alphabetic Index («Том 3») ------------------------------------------- #


def parse_index(path: str) -> tuple[list[dict], str]:
    root = ET.parse(path).getroot()
    version = _text(root.find("version")) or "unknown"
    rows: list[dict] = []

    def walk(node: ET.Element, ancestors: list[str]) -> None:
        title = _text(node.find("title"))
        path_titles = ancestors + [title] if title else ancestors
        # Хвостовой дефис («I63.9-») — маркер неполного кода в указателе (нужны
        # ещё знаки, например латеральность). Срезаем до префикса-рубрики: она
        # есть в Tabular, а уточнение знаков — забота реранкера по neighborhood.
        code = _text(node.find("code")).rstrip("-")
        if code:  # узлы без кода — это чистые <see>/<seeAlso>, пропускаем
            rows.append({"S_NAME": " ".join(path_titles), "ICD-10": code})
        for term in node.findall("term"):
            walk(term, path_titles)

    for letter in root.findall("letter"):
        for main_term in letter.findall("mainTerm"):
            walk(main_term, [])
    return rows, version


# --- запись --------------------------------------------------------------- #


def _dump(rows: list[dict], version: str, out_dir: str, name: str) -> None:
    jsonl_path = os.path.join(out_dir, f"{name}.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta_path = os.path.join(out_dir, f"{name}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {"source": "ICD-10-CM (NCHS/CMS)", "version": version, "count": len(rows)},
            f,
            ensure_ascii=False,
        )
    print(f"    {name}: {len(rows)} записей (версия {version}) -> {jsonl_path}",
          file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="ICD-10-CM XML -> JSONL")
    parser.add_argument("--tabular", required=True, help="icd10cm-tabular-YYYY.xml")
    parser.add_argument("--index", required=True, help="icd10cm-index-YYYY.xml")
    parser.add_argument("--out", default="data/icd10cm", help="каталог для JSONL")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("=== Tabular List (Том 1)", file=sys.stderr)
    tab_rows, tab_ver = parse_tabular(args.tabular)
    _dump(tab_rows, tab_ver, args.out, "tabular")

    print("=== Alphabetic Index (Том 3)", file=sys.stderr)
    idx_rows, idx_ver = parse_index(args.index)
    _dump(idx_rows, idx_ver, args.out, "index")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
