from __future__ import annotations

import json
from pathlib import Path

from bench.datasets import BenchCase, hf_row_to_record, load_split, to_dialogue


def _record(encounter_id: str, subset: str = "virtassist") -> dict:
    return {
        "encounter_id": encounter_id,
        "subset": subset,
        "dialogue": "[doctor] hi there [patient] my head hurts",
        "gold_note": f"gold note {encounter_id}",
        "augmented_note": f"aug note {encounter_id}",
    }


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    return path


def test_hf_row_to_record_maps_columns():
    row = {
        "dataset": "virtscribe",
        "encounter_id": "D2N090",
        "dialogue": "[doctor] hello",
        "note": "the gold note",
        "augmented note": "the augmented note",
    }
    record = hf_row_to_record(row)
    assert record == {
        "encounter_id": "D2N090",
        "subset": "virtscribe",
        "dialogue": "[doctor] hello",
        "gold_note": "the gold note",
        "augmented_note": "the augmented note",
    }


def test_hf_row_to_record_tolerates_missing_augmented():
    row = {
        "dataset": "virtassist",
        "encounter_id": "D2N001",
        "dialogue": "[doctor] hello",
        "note": "gold",
    }
    assert hf_row_to_record(row)["augmented_note"] == ""


def test_load_split_reads_all_cases(tmp_path):
    path = _write_jsonl(tmp_path / "test.jsonl", [_record("D2N002"), _record("D2N001")])
    cases = load_split(path)
    assert [c.encounter_id for c in cases] == ["D2N001", "D2N002"]
    assert cases[0].gold_note == "gold note D2N001"


def test_load_split_subsample_is_deterministic(tmp_path):
    records = [_record(f"D2N{i:03d}") for i in range(10)]
    path = _write_jsonl(tmp_path / "test.jsonl", records)
    first = [c.encounter_id for c in load_split(path, n=4, seed=7)]
    second = [c.encounter_id for c in load_split(path, n=4, seed=7)]
    assert first == second
    assert len(first) == 4


def test_gold_selector():
    case = BenchCase("id", "virtassist", "[doctor] hi", "the gold", "the aug")
    assert case.gold("note") == "the gold"
    assert case.gold("augmented") == "the aug"


def test_to_dialogue_parses_speaker_tags():
    case = BenchCase(
        "id", "virtassist",
        "[doctor] hi there how are you\n[patient] my head hurts doc",
        "gold", "aug",
    )
    dialogue = to_dialogue(case)
    assert [t.role for t in dialogue.turns] == ["doctor", "patient"]
    assert dialogue.turns[0].content == "hi there how are you"
    assert dialogue.turns[1].content == "my head hurts doc"


def test_to_dialogue_keeps_preamble_as_unknown_turn():
    case = BenchCase("id", "virtassist", "intro text [doctor] hi", "gold", "aug")
    dialogue = to_dialogue(case)
    assert dialogue.turns[0].role == "unknown"
    assert dialogue.turns[0].content == "intro text"
    assert dialogue.turns[1].role == "doctor"


def test_to_dialogue_skips_empty_chunks():
    case = BenchCase("id", "virtassist", "[doctor] hi [patient] [doctor] bye", "g", "a")
    dialogue = to_dialogue(case)
    assert [(t.role, t.content) for t in dialogue.turns] == [
        ("doctor", "hi"),
        ("doctor", "bye"),
    ]
