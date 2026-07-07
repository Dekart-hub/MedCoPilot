from __future__ import annotations

import csv
import json

from bench.report import (
    aggregate,
    limitations,
    render_summary,
    select_spot_check,
    write_artifacts,
)


def _case(
    encounter_id: str,
    label: str | None = "normal",
    status: str = "ok",
    subset: str = "virtassist",
    coverage: int = 4,
) -> dict:
    verdict = None
    if status == "ok":
        verdict = {
            "label": label, "coverage": coverage, "correctness": 4,
            "hallucination_free": 5, "hallucinations": [], "rationale": "r",
        }
    return {
        "encounter_id": encounter_id, "subset": subset, "status": status,
        "report_view": {} if status != "failed_extraction" else None,
        "verdict": verdict, "timings_ms": {}, "error": None,
    }


def _config(judge: str = "gpt-4o-mini") -> dict:
    return {
        "run_id": "20260707T120000Z-abc1234", "split": "test", "n": None,
        "seed": 42, "gold": "note", "generator_model": "gpt-4o-mini",
        "judge_model": judge, "data_dir": "data/aci_bench",
        "concurrency": 4, "git_sha": "abc1234", "created_at": "2026-07-07",
    }


def test_aggregate_funnel_and_labels():
    cases = [
        _case("a", label="excellent"),
        _case("b", label="bad", coverage=2),
        _case("c", status="failed_extraction"),
        _case("d", status="failed_judge"),
    ]
    report = aggregate(cases, _config())
    assert report["funnel"] == {
        "total": 4, "extraction_ok": 3, "judged": 2,
        "failed_extraction": 1, "failed_judge": 1,
    }
    assert report["labels"] == {"excellent": 1, "normal": 0, "bad": 1}
    assert report["subscores"]["coverage"] == 3.0
    assert report["by_subset"]["virtassist"]["bad"] == 1


def test_aggregate_with_no_judged_cases():
    report = aggregate([_case("a", status="failed_extraction")], _config())
    assert report["subscores"]["coverage"] is None
    assert report["labels"] == {"excellent": 0, "normal": 0, "bad": 0}


def test_limitations_flags_judge_equals_generator():
    assert any("generator" in line for line in limitations(_config()))
    other = limitations(_config(judge="gpt-4o"))
    assert not any("generator" in line for line in other)


def test_render_summary_mentions_key_numbers():
    report = aggregate([_case("a", label="excellent")], _config())
    text = render_summary(report)
    assert "excellent" in text
    assert "Limitations" in text
    assert report["config"]["run_id"] in text


def test_spot_check_prioritizes_bad_and_caps_at_k():
    cases = (
        [_case(f"bad{i}", label="bad") for i in range(2)]
        + [_case(f"n{i}", label="normal") for i in range(6)]
        + [_case(f"e{i}", label="excellent") for i in range(4)]
    )
    picked = select_spot_check(cases, k=10, seed=1)
    ids = [c["encounter_id"] for c in picked]
    assert len(picked) == 10
    assert "bad0" in ids and "bad1" in ids


def test_spot_check_is_deterministic():
    cases = [_case(f"c{i}", label="normal") for i in range(20)]
    first = [c["encounter_id"] for c in select_spot_check(cases, k=5, seed=3)]
    second = [c["encounter_id"] for c in select_spot_check(cases, k=5, seed=3)]
    assert first == second


def test_write_artifacts(tmp_path):
    cases = [_case("a", label="bad")]
    report = aggregate(cases, _config())
    write_artifacts(tmp_path, report, select_spot_check(cases, k=10))
    assert json.loads((tmp_path / "report.json").read_text())["labels"]["bad"] == 1
    assert "Limitations" in (tmp_path / "summary.md").read_text()
    with open(tmp_path / "spot_check.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["encounter_id"] == "a"
    assert rows[0]["judge_label"] == "bad"
    assert rows[0]["human_label"] == ""
