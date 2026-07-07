#!/usr/bin/env python3
"""Offline SOAP quality benchmark on ACI-Bench-Refined (LLM-as-judge, 3 classes).

Run: split -> ExtractScoredSoap (prod build via DI) -> judge -> report.
Artifacts: runs/soap_bench/<run_id>/{config.json, cases/, report.json,
summary.md, spot_check.csv}.

Usage (needs OPENAI__API_KEY in .env and data/aci_bench from the fetch script):
    uv run python scripts/bench_soap.py --split test --seed 42
    uv run python scripts/bench_soap.py --split validation --n 3     # smoke
    uv run python scripts/bench_soap.py --resume <run_id>            # finish
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain_openai import ChatOpenAI  # noqa: E402

from bench.datasets import load_split  # noqa: E402
from bench.judges import DEFAULT_BENCH_PROMPTS, LlmJudge  # noqa: E402
from bench.report import aggregate, select_spot_check, write_artifacts  # noqa: E402
from bench.runner import BenchRunner  # noqa: E402
from di.container import build_container  # noqa: E402
from shared.prompts import InMemoryPromptStore  # noqa: E402

RUNS_DIR = Path("runs/soap_bench")


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation", "test"),
                        default="test")
    parser.add_argument("--n", type=int, default=None,
                        help="Case sub-sample (defaults to the whole split)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gold", choices=("note", "augmented"), default="note",
                        help="Which reference is the anchor for the judge")
    parser.add_argument("--judge-model", default=None,
                        help="Judge model (else BENCH_JUDGE_MODEL, else the "
                             "generator model)")
    parser.add_argument("--data-dir", default="data/aci_bench")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--resume", default=None, metavar="RUN_ID",
                        help="Finish an existing run (params come from its "
                             "config.json, other flags are ignored)")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    container = await build_container()
    settings = container.settings

    if args.resume:
        out_dir = RUNS_DIR / args.resume
        config_path = out_dir / "config.json"
        if not config_path.exists():
            sys.exit(f"No such run to resume: {config_path} not found")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        print(f"resume {config['run_id']}: params taken from config.json")
    else:
        split_path = Path(args.data_dir) / f"{args.split}.jsonl"
        if not split_path.exists():
            sys.exit(
                f"{split_path} not found — run first: "
                "uv run python scripts/fetch_aci_bench.py"
            )
        now = datetime.now(timezone.utc)
        judge_model = (
            args.judge_model
            or os.environ.get("BENCH_JUDGE_MODEL")
            or settings.openai.model
        )
        config = {
            "run_id": f"{now:%Y%m%dT%H%M%SZ}-{git_sha()}",
            "created_at": now.isoformat(),
            "split": args.split,
            "n": args.n,
            "seed": args.seed,
            "gold": args.gold,
            "generator_model": settings.openai.model,
            "judge_model": judge_model,
            "data_dir": args.data_dir,
            "concurrency": args.concurrency,
            "git_sha": git_sha(),
        }
        out_dir = RUNS_DIR / config["run_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    split_path = Path(config["data_dir"]) / f"{config['split']}.jsonl"
    if not split_path.exists():
        sys.exit(
            f"{split_path} not found — run first: "
            "uv run python scripts/fetch_aci_bench.py"
        )
    cases = load_split(split_path, n=config["n"], seed=config["seed"])
    print(f"{config['run_id']}: {len(cases)} cases from {split_path}")

    judge_llm = ChatOpenAI(
        model=config["judge_model"],
        api_key=settings.openai.api_key,
        base_url=settings.openai.base_url,
        temperature=0.0,
    )
    judge = LlmJudge(judge_llm, InMemoryPromptStore(DEFAULT_BENCH_PROMPTS))
    runner = BenchRunner(
        container.extract_scored_soap,
        judge,
        out_dir / "cases",
        concurrency=config["concurrency"],
        gold=config["gold"],
    )

    results = await runner.run(cases)
    report = aggregate(results, config)
    write_artifacts(out_dir, report, select_spot_check(results, seed=config["seed"]))
    print(f"\nartifacts: {out_dir}\n")
    print((out_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    asyncio.run(main())
