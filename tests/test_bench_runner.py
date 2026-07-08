from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bench.datasets import BenchCase
from bench.judges import CaseVerdict, JudgeError
from bench.runner import (
    STATUS_FAILED_EXTRACTION,
    STATUS_FAILED_JUDGE,
    STATUS_OK,
    BenchRunner,
)
from shared.value_objects import Id
from soap import AssessmentView, ClaimView, NoteView, ReportView


def _view() -> ReportView:
    now = datetime.now(timezone.utc)
    claim = ClaimView(id=Id.new(), claim="c", evidence_text="e", turn_id=Id.new())
    note = NoteView(
        id=Id.new(), subjective=claim, objective=claim,
        assessment=AssessmentView(
            id=Id.new(), claim="a", evidence_text="e", turn_id=Id.new()
        ),
        plan=claim, confidence=0.5,
    )
    return ReportView(id=Id.new(), notes=[note], created_at=now, updated_at=now)


def _verdict() -> CaseVerdict:
    return CaseVerdict(
        label="excellent", coverage=5, correctness=5, hallucination_free=5,
        hallucinations=[], rationale="r",
    )


class FakeExtract:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls = 0
        self._error = error

    async def execute(self, dialogue) -> ReportView:
        self.calls += 1
        if self._error:
            raise self._error
        return _view()


class FakeJudge:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls = 0
        self.golds: list[str] = []
        self._error = error

    async def judge(self, transcript, gold_note, view) -> CaseVerdict:
        self.calls += 1
        self.golds.append(gold_note)
        if self._error:
            raise self._error
        return _verdict()


def _case(encounter_id: str = "D2N001") -> BenchCase:
    return BenchCase(
        encounter_id, "virtassist", "[doctor] hi [patient] head hurts",
        "the gold", "the aug",
    )


def _run(runner: BenchRunner, *cases: BenchCase) -> list[dict]:
    return asyncio.run(runner.run(list(cases)))


def test_ok_case_writes_file_with_verdict(tmp_path: Path):
    runner = BenchRunner(FakeExtract(), FakeJudge(), tmp_path / "cases")
    results = _run(runner, _case())
    assert results[0]["status"] == STATUS_OK
    assert results[0]["verdict"]["label"] == "excellent"
    saved = json.loads((tmp_path / "cases" / "D2N001.json").read_text())
    assert saved["status"] == STATUS_OK
    assert saved["report_view"]["notes"][0]["confidence"] == 0.5
    assert "extraction" in saved["timings_ms"] and "judge" in saved["timings_ms"]


def test_extraction_failure_is_isolated(tmp_path: Path):
    runner = BenchRunner(
        FakeExtract(error=RuntimeError("llm down")), FakeJudge(), tmp_path / "cases"
    )
    results = _run(runner, _case())
    assert results[0]["status"] == STATUS_FAILED_EXTRACTION
    assert results[0]["verdict"] is None
    assert "llm down" in results[0]["error"]


def test_judge_failure_keeps_report_view(tmp_path: Path):
    runner = BenchRunner(
        FakeExtract(), FakeJudge(error=JudgeError("bad output")), tmp_path / "cases"
    )
    results = _run(runner, _case())
    assert results[0]["status"] == STATUS_FAILED_JUDGE
    assert results[0]["report_view"] is not None
    assert results[0]["verdict"] is None


def test_resume_skips_ok_cases(tmp_path: Path):
    first = BenchRunner(FakeExtract(), FakeJudge(), tmp_path / "cases")
    _run(first, _case())
    extract = FakeExtract()
    second = BenchRunner(extract, FakeJudge(), tmp_path / "cases")
    results = _run(second, _case())
    assert extract.calls == 0
    assert results[0]["status"] == STATUS_OK


def test_resume_reruns_failed_cases(tmp_path: Path):
    broken = BenchRunner(
        FakeExtract(error=RuntimeError("boom")), FakeJudge(), tmp_path / "cases"
    )
    _run(broken, _case())
    extract = FakeExtract()
    fixed = BenchRunner(extract, FakeJudge(), tmp_path / "cases")
    results = _run(fixed, _case())
    assert extract.calls == 1
    assert results[0]["status"] == STATUS_OK


# "{truncated" -> JSONDecodeError; "[]" -> valid JSON but non-dict,
# so existing["status"] raises TypeError. Both must trigger a recompute.
@pytest.mark.parametrize("corrupt", ["{truncated", "[]"])
def test_resume_reruns_corrupt_case_files(tmp_path: Path, corrupt: str):
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir(parents=True)
    (cases_dir / "D2N001.json").write_text(corrupt, encoding="utf-8")
    extract = FakeExtract()
    runner = BenchRunner(extract, FakeJudge(), cases_dir)
    results = _run(runner, _case())
    assert extract.calls == 1
    assert results[0]["status"] == STATUS_OK
    saved = json.loads((cases_dir / "D2N001.json").read_text())
    assert saved["status"] == STATUS_OK


def test_gold_kind_is_passed_to_judge(tmp_path: Path):
    judge = FakeJudge()
    runner = BenchRunner(FakeExtract(), judge, tmp_path / "cases", gold="augmented")
    _run(runner, _case())
    assert judge.golds == ["the aug"]
