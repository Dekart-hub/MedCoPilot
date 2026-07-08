"""Judge-vs-human agreement from a filled-in spot_check.csv.

Cohen's kappa is computed by hand (three labels, a dozen pairs) — no point
pulling in sklearn for a single formula.
"""

from __future__ import annotations

from .judges import LABELS


def agreement_stats(pairs: list[tuple[str, str]]) -> dict:
    """Computes judge-vs-human agreement. ``pairs`` = (judge, human)."""
    if not pairs:
        raise ValueError("No filled-in human_label values — nothing to compute")
    for judge, human in pairs:
        if judge not in LABELS or human not in LABELS:
            raise ValueError(f"Unknown label in pair ({judge!r}, {human!r})")
    n = len(pairs)
    confusion = {j: {h: 0 for h in LABELS} for j in LABELS}
    for judge, human in pairs:
        confusion[judge][human] += 1
    agree = sum(confusion[label][label] for label in LABELS)
    po = agree / n
    judge_marginals = {label: sum(confusion[label].values()) for label in LABELS}
    human_marginals = {
        label: sum(confusion[j][label] for j in LABELS) for label in LABELS
    }
    pe = sum(judge_marginals[l] * human_marginals[l] for l in LABELS) / (n * n)
    kappa = 1.0 if pe == 1 else (po - pe) / (1 - pe)
    return {
        "n": n,
        "percent_agreement": round(po, 3),
        "cohen_kappa": round(kappa, 3),
        "confusion": confusion,
    }


def render_agreement(stats: dict) -> str:
    """Markdown agreement summary for the terminal/report."""
    lines = [
        f"n = {stats['n']}",
        f"percent agreement = {stats['percent_agreement']}",
        f"Cohen's kappa = {stats['cohen_kappa']}",
        "",
        "confusion (rows = judge, cols = human):",
        "| judge \\ human | " + " | ".join(LABELS) + " |",
        "|---" * (len(LABELS) + 1) + "|",
    ]
    for judge in LABELS:
        row = " | ".join(str(stats["confusion"][judge][h]) for h in LABELS)
        lines.append(f"| {judge} | {row} |")
    return "\n".join(lines)
