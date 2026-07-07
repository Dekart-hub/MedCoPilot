#!/usr/bin/env python3
"""Judge-vs-human agreement from a filled-in spot_check.csv.

Usage:
    uv run python scripts/bench_agreement.py runs/soap_bench/<run_id>/spot_check.csv

Rows with an empty human_label are skipped.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bench.agreement import agreement_stats, render_agreement  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    args = parser.parse_args()

    with open(args.csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pairs = [
        (row["judge_label"].strip(), row["human_label"].strip())
        for row in rows
        if row["human_label"].strip()
    ]
    skipped = len(rows) - len(pairs)
    if skipped:
        print(f"(rows skipped without human_label: {skipped})\n")
    print(render_agreement(agreement_stats(pairs)))


if __name__ == "__main__":
    main()
