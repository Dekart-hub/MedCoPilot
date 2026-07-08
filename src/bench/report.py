"""Aggregation of run results: report.json, summary.md, spot_check.csv.

Input is the list of case dicts written by ``bench.runner`` (one json per
encounter). This module is pure arithmetic and rendering — no I/O except
``write_artifacts``.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from .judges import LABELS

SUBSCORES = ("coverage", "correctness", "hallucination_free")


def aggregate(cases: list[dict], config: dict) -> dict:
    """Computes the funnel, class distribution and mean subscores."""
    judged = [c for c in cases if c["status"] == "ok"]
    labels = {label: 0 for label in LABELS}
    for case in judged:
        labels[case["verdict"]["label"]] += 1
    subscores: dict[str, float | None] = {}
    for key in SUBSCORES:
        values = [case["verdict"][key] for case in judged]
        subscores[key] = round(sum(values) / len(values), 3) if values else None
    by_subset: dict[str, dict[str, int]] = {}
    for case in judged:
        bucket = by_subset.setdefault(
            case["subset"], {label: 0 for label in LABELS}
        )
        bucket[case["verdict"]["label"]] += 1
    funnel = {
        "total": len(cases),
        "extraction_ok": sum(
            1 for c in cases if c["status"] != "failed_extraction"
        ),
        "judged": len(judged),
        "failed_extraction": sum(
            1 for c in cases if c["status"] == "failed_extraction"
        ),
        "failed_judge": sum(1 for c in cases if c["status"] == "failed_judge"),
    }
    return {
        "config": config,
        "funnel": funnel,
        "labels": labels,
        "subscores": subscores,
        "by_subset": by_subset,
    }


def limitations(config: dict) -> list[str]:
    """Run limitations — printed in every summary.md."""
    lines = [
        "Speaker tags in ACI-Bench are occasionally ASR-swapped; we do not fix them.",
        "Flat 4-section SOAP is graded against a richer reference note; the judge "
        "is instructed not to penalize format, but residual noise remains.",
        "Small split sizes (test = 20 encounters): confidence intervals are wide.",
    ]
    if config["judge_model"] == config["generator_model"]:
        lines.insert(
            0,
            f"Judge model equals generator model ({config['judge_model']}): "
            "self-preference bias is possible.",
        )
    return lines


def render_summary(report: dict) -> str:
    """Human-readable markdown report for the run."""
    config, funnel = report["config"], report["funnel"]
    judged = funnel["judged"] or 1  # guard against division by zero in percentages
    lines = [
        f"# SOAP bench — {config['run_id']}",
        "",
        f"- split: `{config['split']}`, n={config['n'] or 'all'}, "
        f"seed={config['seed']}, gold=`{config['gold']}`",
        f"- generator: `{config['generator_model']}`, judge: `{config['judge_model']}`",
        f"- git: `{config['git_sha']}`, created: {config['created_at']}",
        "",
        "## Funnel",
        "",
        f"- total cases: {funnel['total']}",
        f"- extraction ok: {funnel['extraction_ok']} "
        f"(failed: {funnel['failed_extraction']})",
        f"- judged: {funnel['judged']} (judge failed: {funnel['failed_judge']})",
        "",
        "## Labels",
        "",
    ]
    for label in LABELS:
        count = report["labels"][label]
        lines.append(f"- {label}: {count} ({100 * count / judged:.0f}%)")
    lines += ["", "## Mean subscores (1-5)", ""]
    for key in SUBSCORES:
        value = report["subscores"][key]
        lines.append(f"- {key}: {value if value is not None else 'n/a'}")
    lines += ["", "## By subset", ""]
    for subset, counts in sorted(report["by_subset"].items()):
        parts = ", ".join(f"{label}: {counts[label]}" for label in LABELS)
        lines.append(f"- {subset}: {parts}")
    lines += ["", "## Limitations", ""]
    lines += [f"- {line}" for line in limitations(config)]
    return "\n".join(lines) + "\n"


def select_spot_check(cases: list[dict], k: int = 10, seed: int = 42) -> list[dict]:
    """Stratified sub-sample for manual review: all bad cases first."""
    judged = [c for c in cases if c["status"] == "ok"]
    rng = random.Random(seed)
    by_label = {
        label: [c for c in judged if c["verdict"]["label"] == label]
        for label in LABELS
    }
    for group in by_label.values():
        rng.shuffle(group)
    picked = by_label["bad"][:k]
    pools = [by_label["normal"], by_label["excellent"]]
    while len(picked) < k and any(pools):
        for pool in pools:
            if pool and len(picked) < k:
                picked.append(pool.pop())
    return sorted(picked, key=lambda c: c["encounter_id"])


def write_artifacts(out_dir: Path, report: dict, spot_cases: list[dict]) -> None:
    """Writes report.json, summary.md and spot_check.csv into the run directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "summary.md").write_text(render_summary(report), encoding="utf-8")
    with open(out_dir / "spot_check.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "encounter_id", "judge_label", "rationale", "human_label", "comment",
            ],
        )
        writer.writeheader()
        for case in spot_cases:
            writer.writerow(
                {
                    "encounter_id": case["encounter_id"],
                    "judge_label": case["verdict"]["label"],
                    "rationale": case["verdict"]["rationale"],
                    "human_label": "",
                    "comment": "",
                }
            )
