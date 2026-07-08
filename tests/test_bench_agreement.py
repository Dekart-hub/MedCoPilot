from __future__ import annotations

import pytest

from bench.agreement import agreement_stats, render_agreement


def test_perfect_agreement_gives_kappa_one():
    pairs = [("bad", "bad"), ("normal", "normal"), ("excellent", "excellent")]
    stats = agreement_stats(pairs)
    assert stats["percent_agreement"] == 1.0
    assert stats["cohen_kappa"] == 1.0


def test_known_kappa_value():
    # n=4, po=0.75; marginals judge (bad 2, normal 1, exc 1),
    # human (bad 1, normal 2, exc 1) -> pe=5/16 -> kappa=0.636.
    pairs = [
        ("bad", "bad"), ("bad", "normal"),
        ("normal", "normal"), ("excellent", "excellent"),
    ]
    stats = agreement_stats(pairs)
    assert stats["percent_agreement"] == 0.75
    assert stats["cohen_kappa"] == 0.636
    assert stats["confusion"]["bad"]["normal"] == 1


def test_empty_pairs_raise():
    with pytest.raises(ValueError):
        agreement_stats([])


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        agreement_stats([("bad", "great")])


def test_render_agreement_mentions_kappa():
    text = render_agreement(agreement_stats([("bad", "bad")]))
    assert "kappa" in text.lower()
