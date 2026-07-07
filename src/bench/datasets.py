"""ACI-Bench-Refined dataset: HF rows → jsonl records → BenchCase → Dialogue.

Splits are downloaded by ``scripts/fetch_aci_bench.py`` into ``data/aci_bench/``
(never committed — AGPL). This module only reads and converts to the domain.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dialogue import Dialogue, DialogueTurn
from shared.value_objects import Id

# The HF dataset column name contains a space: "augmented note".
_HF_AUGMENTED_COLUMN = "augmented note"

# Speaker tags in the dialogue: "[doctor] ... [patient] ...".
_TAG_RE = re.compile(r"\[([a-z_0-9]+)\]", re.IGNORECASE)


def hf_row_to_record(row: dict) -> dict:
    """HF parquet row → flat jsonl benchmark record."""
    return {
        "encounter_id": str(row["encounter_id"]),
        "subset": row["dataset"],
        "dialogue": row["dialogue"],
        "gold_note": row["note"],
        "augmented_note": row.get(_HF_AUGMENTED_COLUMN, "") or "",
    }


@dataclass(frozen=True, slots=True)
class BenchCase:
    """One dataset encounter: transcript + two references."""

    encounter_id: str
    subset: str
    dialogue_text: str
    gold_note: str
    augmented_note: str

    def gold(self, kind: str) -> str:
        """Reference for the judge: ``"augmented"`` → refined note, else original."""
        return self.augmented_note if kind == "augmented" else self.gold_note


def load_split(path: Path, n: int | None = None, seed: int = 42) -> list[BenchCase]:
    """Reads a jsonl split; with ``n``, a deterministic sub-sample by seed."""
    cases: list[BenchCase] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            cases.append(
                BenchCase(
                    encounter_id=rec["encounter_id"],
                    subset=rec["subset"],
                    dialogue_text=rec["dialogue"],
                    gold_note=rec["gold_note"],
                    augmented_note=rec.get("augmented_note", ""),
                )
            )
    cases.sort(key=lambda c: c.encounter_id)
    if n is not None and n < len(cases):
        cases = random.Random(seed).sample(cases, n)
        cases.sort(key=lambda c: c.encounter_id)
    return cases


def to_dialogue(case: BenchCase) -> Dialogue:
    """Builds a domain ``Dialogue`` from the ``[role]`` markup of the transcript.

    Text before the first tag (if any) becomes an ``unknown``-role turn;
    empty chunks between tags are skipped. The dataset's swapped ASR tags
    are left as-is — that is a documented property of the dataset.
    """
    now = datetime.now(timezone.utc)
    parts = _TAG_RE.split(case.dialogue_text)
    turns: list[DialogueTurn] = []
    preamble = parts[0].strip()
    if preamble:
        turns.append(
            DialogueTurn(id=Id.new(), role="unknown", content=preamble, timestamp=now)
        )
    for role, content in zip(parts[1::2], parts[2::2]):
        content = content.strip()
        if not content:
            continue
        turns.append(
            DialogueTurn(id=Id.new(), role=role.lower(), content=content, timestamp=now)
        )
    return Dialogue(id=Id.new(), turns=turns, created_at=now)
