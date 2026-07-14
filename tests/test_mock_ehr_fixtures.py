from __future__ import annotations

import json
from pathlib import Path

from ehr import FIXTURE_PHASE_SYSTEM, POST_VISIT_PHASE, PRE_VISIT_PHASE, WHO_ICD10_SYSTEM

FIXTURES = Path(__file__).resolve().parents[1] / "mock_ehr" / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _resources(bundle: dict) -> list[dict]:
    return [entry["resource"] for entry in bundle["entry"]]


def test_manifest_references_exist_in_separate_fixture_phases():
    manifest = _load("manifest.json")
    pre = _resources(_load("pre-visit-bundle.json"))
    post = _resources(_load("post-visit-bundle.json"))
    case = manifest["cases"][0]

    pre_refs = {f"{resource['resourceType']}/{resource['id']}" for resource in pre}
    post_refs = {f"{resource['resourceType']}/{resource['id']}" for resource in post}

    assert case["patient_ref"] in pre_refs
    assert case["encounter_ref"] in pre_refs
    assert set(case["pre_visit"]["condition_refs"]) <= pre_refs
    assert case["post_visit"]["condition_ref"] in post_refs
    assert case["post_visit"]["condition_ref"] not in pre_refs


def test_clinical_resources_are_phase_tagged_and_conditions_use_who_icd10():
    for filename, phase in (
        ("pre-visit-bundle.json", PRE_VISIT_PHASE),
        ("post-visit-bundle.json", POST_VISIT_PHASE),
    ):
        resources = _resources(_load(filename))
        for resource in resources:
            if resource["resourceType"] in {
                "Condition",
                "AllergyIntolerance",
                "MedicationRequest",
                "Observation",
            }:
                assert {
                    "system": FIXTURE_PHASE_SYSTEM,
                    "code": phase,
                } in resource["meta"]["tag"]
            if resource["resourceType"] == "Condition":
                codings = resource["code"]["coding"]
                assert any(coding.get("system") == WHO_ICD10_SYSTEM for coding in codings)
