#!/usr/bin/env python3
"""Materialise a fuller ICD-10 dictionary for the BM25 coder (T10).

Source (documented, public, no auth): the CMS ICD-10-CM code descriptions
release — the tabular-order code list published by the US Centers for Medicare &
Medicaid Services / NCHS:

    https://www.cms.gov/medicare/coding-billing/icd-10-codes

The release ships a zip whose ``icd10cm-codes-*.txt`` lists one code per line as
``<code> <description>``. We parse it into the same ``[{"code", "name"}, ...]``
JSON schema the package's bundled sample uses, inserting the conventional dot
after the third character so codes match the WHO ICD-10 browser reference URL.

Note: ICD-10-CM is the US clinical modification of WHO ICD-10; category-level
codes line up, but some fine-grained CM subcodes have no WHO browser page. The
unit tests deliberately use the bundled sample (``src/icd/icd10_sample.json``),
never this script or the network.

Usage:
    python scripts/fetch_icd.py [--out data/icd10.json]
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

# Pin a specific documented release for reproducibility; bump as CMS publishes.
_SOURCE_URL = "https://www.cms.gov/files/zip/2024-code-descriptions-tabular-order.zip"
# Keep in lockstep with the pinned URL: this string becomes the
# ``classifier_version`` stamped on every resolution made over this dictionary.
_VERSION = "icd10cm-2024-cms"
_DEFAULT_OUT = Path("data/icd10.json")


def _dotted(code: str) -> str:
    """Insert the ICD-10 category dot: ``J189`` -> ``J18.9`` (``J18`` unchanged)."""
    return f"{code[:3]}.{code[3:]}" if len(code) > 3 else code


def _parse(text: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for line in text.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            code, name = parts
            entries.append({"code": _dotted(code), "name": name.strip()})
    return entries


def fetch(url: str = _SOURCE_URL) -> list[dict[str, str]]:
    with urllib.request.urlopen(url) as response:  # noqa: S310 - documented CMS host
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        codes_file = next(n for n in archive.namelist() if n.endswith(".txt") and "codes" in n)
        text = archive.read(codes_file).decode("utf-8", errors="replace")
    return _parse(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT, help="output JSON path")
    args = parser.parse_args()

    entries = fetch()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {"source": _SOURCE_URL, "version": _VERSION, "count": len(entries)}
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(entries)} ICD-10 codes to {args.out} (version {_VERSION})")


if __name__ == "__main__":
    main()
