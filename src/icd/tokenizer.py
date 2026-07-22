"""Clinical tokenizer shared by the BM25 coder and resolver.

Ported from the retrieval work on the classifier branch. Three deliberate
choices, each measured on real coding data:

* Snowball stemming, so "diabetic"/"diabetes" and singular/plural forms meet.
* Clinical abbreviation expansion ("t2dm", "copd", ...) — doctors dictate in
  abbreviations, catalog titles never do.
* The stopword list drops function words and ICD jargon (NEC/NOS,
  "unspecified") but deliberately KEEPS the qualifiers ``with`` / ``without`` /
  ``no`` / ``not``: they are what distinguishes 4th–5th code characters
  ("with complications" vs "without complications").
"""

from __future__ import annotations

import re

import snowballstemmer  # type: ignore[import-untyped]  # ships no stubs

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "due",
        "for",
        "from",
        "in",
        "nec",
        "nos",
        "of",
        "on",
        "or",
        "other",
        "the",
        "to",
        "unspecified",
    }
)

# Expansions map a dictated abbreviation onto the catalog's own wording; only
# unambiguous, standalone clinical abbreviations belong here.
_ABBREVIATIONS: dict[str, tuple[str, ...]] = {
    "afib": ("atrial", "fibrillation"),
    "cad": ("coronary", "artery", "disease"),
    "chf": ("congestive", "heart", "failure"),
    "ckd": ("chronic", "kidney", "disease"),
    "copd": ("chronic", "obstructive", "pulmonary", "disease"),
    "gerd": ("gastroesophageal", "reflux", "disease"),
    "htn": ("hypertension",),
    "ibs": ("irritable", "bowel", "syndrome"),
    "mi": ("myocardial", "infarction"),
    "t1dm": ("type", "1", "diabetes", "mellitus"),
    "t2dm": ("type", "2", "diabetes", "mellitus"),
    "tia": ("transient", "ischemic", "attack"),
    "uri": ("upper", "respiratory", "infection"),
    "uti": ("urinary", "tract", "infection"),
}

_stemmer = snowballstemmer.stemmer("english")


def tokenize(text: str) -> list[str]:
    """Normalise ``text`` into stemmed content tokens for BM25."""
    expanded: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        expanded.extend(_ABBREVIATIONS.get(token, (token,)))
    stemmed: list[str] = _stemmer.stemWords(
        [token for token in expanded if token not in _STOPWORDS]
    )
    return stemmed
