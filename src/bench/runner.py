"""Run orchestration: pipeline + judge, one file per case, resume.

Each case is isolated: any exception moves it to a failed_* status and the run
continues. Successful cases (files with status ``ok``) are not recomputed on a
rerun — LLM calls are expensive.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

from soap import ExtractScoredSoap, ReportView

from .datasets import BenchCase, to_dialogue
from .judges import LlmJudge

logger = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_FAILED_EXTRACTION = "failed_extraction"
STATUS_FAILED_JUDGE = "failed_judge"


def _view_to_dict(view: ReportView) -> dict:
    # UUID/datetime inside dataclasses -> strings; round-trip gives a clean dict.
    return json.loads(json.dumps(asdict(view), default=str))


class BenchRunner:
    """Runs cases through ExtractScoredSoap and the judge, writes cases/*.json."""

    def __init__(
        self,
        extract: ExtractScoredSoap,
        judge: LlmJudge,
        cases_dir: Path,
        concurrency: int = 4,
        gold: str = "note",
    ) -> None:
        self._extract = extract
        self._judge = judge
        self._cases_dir = cases_dir
        self._concurrency = concurrency
        self._gold = gold

    async def run(self, cases: list[BenchCase]) -> list[dict]:
        self._cases_dir.mkdir(parents=True, exist_ok=True)
        semaphore = asyncio.Semaphore(self._concurrency)

        async def one(case: BenchCase) -> dict:
            async with semaphore:
                return await self._run_case(case)

        return list(await asyncio.gather(*(one(case) for case in cases)))

    async def _run_case(self, case: BenchCase) -> dict:
        path = self._cases_dir / f"{case.encounter_id}.json"
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing["status"] == STATUS_OK:
                    return existing  # resume: don't recompute a successful case
            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                UnicodeDecodeError,
                OSError,
            ):
                # corrupt/truncated file (write is not atomic) = recompute
                logger.warning("corrupt case file, recomputing: %s", case.encounter_id)

        result: dict = {
            "encounter_id": case.encounter_id,
            "subset": case.subset,
            "status": STATUS_OK,
            "report_view": None,
            "verdict": None,
            "timings_ms": {},
            "error": None,
        }

        started = time.monotonic()
        try:
            view = await self._extract.execute(to_dialogue(case))
        except Exception as exc:
            logger.exception("extraction failed: %s", case.encounter_id)
            result["status"] = STATUS_FAILED_EXTRACTION
            result["error"] = repr(exc)
            self._write(path, result)
            return result
        result["timings_ms"]["extraction"] = int((time.monotonic() - started) * 1000)
        result["report_view"] = _view_to_dict(view)

        started = time.monotonic()
        try:
            verdict = await self._judge.judge(
                case.dialogue_text, case.gold(self._gold), view
            )
        except Exception as exc:
            logger.exception("judge failed: %s", case.encounter_id)
            result["status"] = STATUS_FAILED_JUDGE
            result["error"] = repr(exc)
            self._write(path, result)
            return result
        result["timings_ms"]["judge"] = int((time.monotonic() - started) * 1000)
        result["verdict"] = verdict.model_dump()

        self._write(path, result)
        return result

    @staticmethod
    def _write(path: Path, result: dict) -> None:
        # atomic: write to a sibling temp file and swap — resume won't see a partial write
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp, path)
