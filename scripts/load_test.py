#!/usr/bin/env python3
"""Load-test harness for the E2E extraction pipeline  [#7/NFR-1].

Drives the REST API end to end — for each measured request it creates a fresh
dialogue (``POST /dialogues``) and then times the expensive extraction+scoring
step (``POST /dialogues/{id}/report``). Extraction is idempotent per dialogue, so
a fresh dialogue every iteration is what forces a real LLM call rather than a
cached hit. Only the report step is timed; the dialogue creation is setup.

It runs a warmup burst, then N measured requests at a chosen concurrency, and
prints P50/P95/P99 (plus count/mean/max) for the report latency, stating the
result against the NFR-1 target of P99 <= 5s. Not pytest and not part of CI.

The authoritative measurement is run later against the full compose stack
(``docker compose --profile gpu up``). See the README "Load test" section.

    LOAD_BASE_URL=http://localhost:8000 \
        uv run python scripts/load_test.py --requests 30 --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import time
from dataclasses import dataclass

import httpx

FIXTURES: list[list[tuple[str, str]]] = [
    [
        ("doctor", "What brings you in today?"),
        (
            "patient",
            "I've had a fever and a productive cough for three days, and I feel short of breath.",
        ),
        (
            "doctor",
            "Any chest pain? I hear crackles in the right lower lobe, temp is 38.6.",
        ),
        ("patient", "Yes, it hurts when I breathe deeply."),
        (
            "doctor",
            "This looks like community-acquired pneumonia. I'll start amoxicillin "
            "and order a chest X-ray.",
        ),
    ],
    [
        (
            "doctor",
            "Your home readings have been running high — what numbers are you seeing?",
        ),
        (
            "patient",
            "Most mornings it's around 150 over 95, and I get headaches at the back of my head.",
        ),
        ("doctor", "Any chest pain, palpitations, or visual changes?"),
        (
            "patient",
            "No chest pain, but occasional palpitations. My father had a stroke at 60.",
        ),
        (
            "doctor",
            "In clinic your blood pressure is 152 over 96. This is stage 2 "
            "hypertension; I'll start amlodipine 5 mg.",
        ),
    ],
    [
        ("doctor", "What's going on with your back?"),
        (
            "patient",
            "I lifted a heavy box at work four days ago and my lower back has ached ever since.",
        ),
        (
            "doctor",
            "Does the pain travel down your legs? Any numbness or bladder trouble?",
        ),
        ("patient", "It stays in the lower back, no numbness, and my bladder is fine."),
        (
            "doctor",
            "This is a mechanical lumbar strain. Take ibuprofen with food and stay gently active.",
        ),
    ],
]


@dataclass(frozen=True)
class Config:
    """Everything the run needs, resolved from CLI args over env defaults."""

    base_url: str
    requests: int
    concurrency: int
    warmup: int
    threshold: float
    timeout: float
    patient_id: str | None


def _parse_config() -> Config:
    parser = argparse.ArgumentParser(description="E2E extraction load test (P99 <= 5s target).")
    parser.add_argument(
        "--base-url", default=os.environ.get("LOAD_BASE_URL", "http://localhost:8000")
    )
    parser.add_argument(
        "-n", "--requests", type=int, default=int(os.environ.get("LOAD_REQUESTS", "30"))
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=int(os.environ.get("LOAD_CONCURRENCY", "4")),
    )
    parser.add_argument("-w", "--warmup", type=int, default=int(os.environ.get("LOAD_WARMUP", "2")))
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.environ.get("LOAD_P99_THRESHOLD", "5.0")),
    )
    parser.add_argument(
        "--timeout", type=float, default=float(os.environ.get("LOAD_TIMEOUT", "120.0"))
    )
    parser.add_argument("--patient-id", default=os.environ.get("LOAD_PATIENT_ID") or None)
    args = parser.parse_args()
    return Config(
        base_url=args.base_url,
        requests=args.requests,
        concurrency=args.concurrency,
        warmup=args.warmup,
        threshold=args.threshold,
        timeout=args.timeout,
        patient_id=args.patient_id,
    )


def _dialogue_payload(turns: list[tuple[str, str]]) -> dict[str, object]:
    return {"turns": [{"speaker": speaker, "text": text} for speaker, text in turns]}


async def _create_dialogue(client: httpx.AsyncClient, turns: list[tuple[str, str]]) -> str:
    response = await client.post("/dialogues", json=_dialogue_payload(turns))
    response.raise_for_status()
    dialogue_id: str = response.json()["id"]
    return dialogue_id


async def _extract_report(
    client: httpx.AsyncClient, dialogue_id: str, patient_id: str | None
) -> None:
    params = {"patient_id": patient_id} if patient_id else None
    response = await client.post(f"/dialogues/{dialogue_id}/report", params=params)
    response.raise_for_status()


async def _measure_once(
    client: httpx.AsyncClient, turns: list[tuple[str, str]], patient_id: str | None
) -> float:
    """Create a fresh dialogue, then return the report-extraction latency in seconds."""
    dialogue_id = await _create_dialogue(client, turns)
    start = time.perf_counter()
    await _extract_report(client, dialogue_id, patient_id)
    return time.perf_counter() - start


async def _run_batch(
    client: httpx.AsyncClient, config: Config, count: int
) -> tuple[list[float], int]:
    """Fire ``count`` measured requests at the configured concurrency."""
    semaphore = asyncio.Semaphore(config.concurrency)

    async def worker(index: int) -> float | None:
        async with semaphore:
            turns = FIXTURES[index % len(FIXTURES)]
            try:
                return await _measure_once(client, turns, config.patient_id)
            except Exception as exc:
                print(f"  request {index} failed: {exc}")
                return None

    results = await asyncio.gather(*(worker(i) for i in range(count)))
    latencies = [value for value in results if value is not None]
    return latencies, count - len(latencies)


def _percentile(sorted_latencies: list[float], pct: float) -> float:
    """Nearest-rank percentile over an already-sorted list."""
    rank = math.ceil(pct / 100 * len(sorted_latencies))
    index = min(max(rank, 1), len(sorted_latencies)) - 1
    return sorted_latencies[index]


def _report(config: Config, latencies: list[float], failed: int) -> bool:
    """Print the latency summary and return whether the run passes NFR-1."""
    print("\nE2E report-extraction latency")
    print(f"  base URL:    {config.base_url}")
    print(f"  measured:    {config.requests} requests, concurrency {config.concurrency}")
    print(f"  succeeded:   {len(latencies)}")
    print(f"  failed:      {failed}")
    if not latencies:
        print("  no successful requests — nothing to measure")
        return False
    ordered = sorted(latencies)
    print(f"  mean:        {sum(ordered) / len(ordered):.3f} s")
    print(f"  p50:         {_percentile(ordered, 50):.3f} s")
    print(f"  p95:         {_percentile(ordered, 95):.3f} s")
    p99 = _percentile(ordered, 99)
    print(f"  p99:         {p99:.3f} s")
    print(f"  max:         {ordered[-1]:.3f} s")
    passed = failed == 0 and p99 <= config.threshold
    verdict = "PASS" if passed else "FAIL"
    print(f"  P99 threshold {config.threshold:.3f} s  ->  {verdict}")
    return passed


async def _run(config: Config) -> int:
    async with httpx.AsyncClient(base_url=config.base_url, timeout=config.timeout) as client:
        if config.warmup:
            warm, warm_failed = await _run_batch(client, config, config.warmup)
            print(f"warmup: {len(warm)} ok / {config.warmup} ({warm_failed} failed)")
        latencies, failed = await _run_batch(client, config, config.requests)
    return 0 if _report(config, latencies, failed) else 1


def main() -> int:
    return asyncio.run(_run(_parse_config()))


if __name__ == "__main__":
    raise SystemExit(main())
