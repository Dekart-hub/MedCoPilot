#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "mock_ehr" / "fixtures"
PHASE_FILES = {
    "pre-visit": FIXTURES / "pre-visit-bundle.json",
    "post-visit": FIXTURES / "post-visit-bundle.json",
}


def _wait_for_server(client: httpx.Client, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = client.get("metadata")
            if response.is_success:
                return
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(2)
    detail = f": {last_error}" if last_error else ""
    raise RuntimeError(f"Mock EHR did not become ready{detail}")


def _load_bundle(path: Path) -> dict:
    with path.open(encoding="utf-8") as fixture:
        return json.load(fixture)


def _apply_bundle(client: httpx.Client, phase: str) -> None:
    path = PHASE_FILES[phase]
    response = client.post(
        "",
        json=_load_bundle(path),
        headers={
            "Accept": "application/fhir+json",
            "Content-Type": "application/fhir+json",
        },
    )
    if not response.is_success:
        raise RuntimeError(
            f"Failed to apply {path.name}: HTTP {response.status_code} "
            f"{response.text[:500]}"
        )
    payload = response.json()
    if payload.get("resourceType") != "Bundle":
        raise RuntimeError(f"Unexpected transaction response for {path.name}")
    print(f"Applied {phase} fixtures ({len(payload.get('entry', []))} entries)")


def _verify_pre_visit(client: httpx.Client) -> None:
    patient = client.get("Patient/mock-patient-001")
    encounter = client.get("Encounter/mock-encounter-001")
    conditions = client.get(
        "Condition", params={"patient": "mock-patient-001"}
    )
    for response in (patient, encounter, conditions):
        response.raise_for_status()

    condition_refs = {
        f"Condition/{entry['resource']['id']}"
        for entry in conditions.json().get("entry", [])
    }
    expected = "Condition/mock-condition-covid-001"
    if expected not in condition_refs:
        raise RuntimeError(f"Seed verification failed: {expected} is missing")
    print("Verified patient, encounter, and pre-visit Condition resources")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load deterministic FHIR R4 fixtures into the external mock EHR"
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("MOCK_EHR_BASE_URL", "http://localhost:8080/fhir"),
    )
    parser.add_argument(
        "--phase",
        choices=("pre-visit", "post-visit", "all"),
        default="pre-visit",
        help="Pre-visit is the safe default; post-visit contains the gold diagnosis",
    )
    parser.add_argument("--wait-seconds", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    base_url = f"{args.base_url.rstrip('/')}/"
    phases = ("pre-visit", "post-visit") if args.phase == "all" else (args.phase,)
    try:
        with httpx.Client(base_url=base_url, timeout=20.0) as client:
            _wait_for_server(client, args.wait_seconds)
            for phase in phases:
                _apply_bundle(client, phase)
            _verify_pre_visit(client)
    except (httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
        print(f"mock-ehr seed failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
