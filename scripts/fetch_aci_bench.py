#!/usr/bin/env python3
"""Downloads ClinicianFOCUS/ACI-Bench-Refined into data/aci_bench/*.jsonl.

Source is the public HF parquet export (no API key needed). Expected split
sizes: train 177, validation 10, test 20.

Usage:
    uv run python scripts/fetch_aci_bench.py [--out data/aci_bench]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

import httpx
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bench.datasets import hf_row_to_record  # noqa: E402

DATASET = "ClinicianFOCUS/ACI-Bench-Refined"
SPLITS = ("train", "validation", "test")
PARQUET_URL = (
    "https://huggingface.co/api/datasets/{dataset}/parquet/default/{split}/0.parquet"
)


def fetch_split(client: httpx.Client, split: str) -> list[dict]:
    url = PARQUET_URL.format(dataset=DATASET, split=split)
    response = client.get(url, follow_redirects=True)
    response.raise_for_status()
    table = pq.read_table(io.BytesIO(response.content))
    return [hf_row_to_record(row) for row in table.to_pylist()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/aci_bench")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with httpx.Client(timeout=120) as client:
        for split in SPLITS:
            records = fetch_split(client, split)
            path = os.path.join(args.out, f"{split}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{split}: {len(records)} encounters -> {path}")


if __name__ == "__main__":
    main()
