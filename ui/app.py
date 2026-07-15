"""Streamlit demo UI for the MedCoPilot baseline.

A manual, demo-only interface to exercise the baseline REST API (T13): paste a
doctor-patient dialogue, optionally pin a ``patient_id``, and see the extracted
``SoapReport`` -- the four S/O/A/P sections, the ICD code on Assessment claims,
per-note confidence, and every claim linked back to the dialogue turn it cites.

This is **out of scope** for the baseline DoD; it is a thin client over the API
and deliberately kept simple. It talks HTTP only -- it imports nothing from
``src`` so it stays decoupled from the service internals.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

DEFAULT_API_URL = os.environ.get("MEDCOPILOT_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 120.0

SECTION_TITLES = {
    "subjective": "Subjective",
    "objective": "Objective",
    "assessment": "Assessment",
    "plan": "Plan",
}

EXAMPLE_DIALOGUE = (
    "doctor: What brings you in today?\n"
    "patient: I've had a fever and a bad cough for three days.\n"
    "doctor: Any chest pain or shortness of breath?\n"
    "patient: Some tightness when I breathe deeply.\n"
    "doctor: Your temperature is 38.6 and I hear crackles in the right lung.\n"
    "doctor: This looks like community-acquired pneumonia; let's start antibiotics."
)


def parse_turns(text: str) -> list[dict[str, str]]:
    """Parse ``speaker: text`` lines into API turn payloads, in order."""
    turns: list[dict[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        speaker, separator, utterance = stripped.partition(":")
        if separator:
            turns.append({"speaker": speaker.strip(), "text": utterance.strip()})
        else:
            turns.append({"speaker": "unknown", "text": stripped})
    return turns


def create_dialogue(base_url: str, turns: list[dict[str, str]]) -> str:
    """POST the turns to ``/dialogues`` and return the new dialogue id."""
    response = httpx.post(
        f"{base_url}/dialogues",
        json={"turns": turns},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return str(response.json()["id"])


def extract_report(base_url: str, dialogue_id: str, patient_id: str) -> dict[str, Any]:
    """POST to ``/dialogues/{id}/report`` and return the serialized SoapReport."""
    params = {"patient_id": patient_id} if patient_id else None
    response = httpx.post(
        f"{base_url}/dialogues/{dialogue_id}/report",
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def describe_http_error(error: httpx.HTTPStatusError) -> str:
    """Turn a non-2xx response into a readable one-line message."""
    try:
        detail = error.response.json().get("detail", error.response.text)
    except ValueError:
        detail = error.response.text
    return f"API returned {error.response.status_code}: {detail}"


def find_source_turn(quote: str | None, turns: list[dict[str, str]]) -> int | None:
    """Locate which entered turn a quote came from (1-based), or ``None``.

    The API returns opaque turn ids, so the demo re-links a citation to the
    dialogue you typed by matching its verbatim quote against each turn.
    """
    if not quote:
        return None
    needle = " ".join(quote.lower().split())
    for index, turn in enumerate(turns, start=1):
        haystack = " ".join(turn["text"].lower().split())
        if needle and needle in haystack:
            return index
    return None


def render_citation(citation: dict[str, Any], turns: list[dict[str, str]]) -> None:
    """Render one claim->turn link, resolving it back to the entered turn."""
    quote = citation.get("quote")
    source = find_source_turn(quote, turns)
    if source is not None:
        turn = turns[source - 1]
        st.markdown(f"↳ **Turn {source}** — _{turn['speaker']}_: {turn['text']}")
    else:
        st.markdown(f"↳ source turn `{citation['turn_id'][:8]}…`")
    if quote:
        st.caption(f'cited: "{quote}"')


def render_icd(icd: dict[str, str] | None) -> None:
    """Render the ICD coding of an Assessment claim as a linked code + name."""
    if icd is None:
        st.caption("ICD: not coded")
        return
    st.markdown(f"**ICD [{icd['code']}]({icd['classifier_url']})** — {icd['name']}")


def render_claim(claim: dict[str, Any], turns: list[dict[str, str]], *, assessment: bool) -> None:
    """Render a single claim: its text, optional ICD coding, and citations."""
    st.markdown(f"- {claim['text']}")
    if assessment:
        render_icd(claim.get("icd"))
    for citation in claim["citations"]:
        render_citation(citation, turns)


def render_section(
    section_key: str, claims: list[dict[str, Any]], turns: list[dict[str, str]]
) -> None:
    """Render one SOAP section and all of its claims."""
    st.markdown(f"#### {SECTION_TITLES.get(section_key, section_key.title())}")
    if not claims:
        st.caption("_(no claims)_")
        return
    for claim in claims:
        render_claim(claim, turns, assessment=section_key == "assessment")


def render_confidence(confidence: float | None) -> None:
    """Render a note's confidence as a metric, tolerating an unset score."""
    st.metric("Confidence", "n/a" if confidence is None else f"{confidence:.2f}")


def render_report(report: dict[str, Any], turns: list[dict[str, str]]) -> None:
    """Render the full report: every note, its confidence and S/O/A/P sections."""
    st.subheader("SOAP report")
    st.caption(f"report id: {report['id']} · {len(report['notes'])} note(s)")
    for position, note in enumerate(report["notes"], start=1):
        st.markdown(f"### Note {position}")
        render_confidence(note.get("confidence"))
        sections = note["sections"]
        for section_key in SECTION_TITLES:
            render_section(section_key, sections.get(section_key, []), turns)
    with st.expander("Raw JSON"):
        st.json(report)


def run_extraction(base_url: str, turns: list[dict[str, str]], patient_id: str) -> None:
    """Create the dialogue, extract its report, and render it (or an error)."""
    try:
        with st.spinner("Creating dialogue and extracting report…"):
            dialogue_id = create_dialogue(base_url, turns)
            report = extract_report(base_url, dialogue_id, patient_id)
    except httpx.HTTPStatusError as error:
        st.error(describe_http_error(error))
    except httpx.RequestError as error:
        st.error(f"Could not reach the API at {base_url} — is it running? ({error})")
    else:
        st.success(f"Extracted report for dialogue {dialogue_id}")
        render_report(report, turns)


def sidebar_base_url() -> str:
    """Render the sidebar and return the configured API base URL."""
    with st.sidebar:
        st.header("Settings")
        base_url = st.text_input("API base URL", value=DEFAULT_API_URL)
        st.caption("Start the baseline API first: `make run` (defaults to :8000).")
    return base_url.rstrip("/")


def main() -> None:
    """Entry point: collect inputs and, on submit, extract and render a report."""
    st.set_page_config(page_title="MedCoPilot demo", page_icon="🩺")
    st.title("MedCoPilot — SOAP extraction demo")
    st.write("Paste a dialogue as `speaker: text` lines, then extract its SOAP report.")

    base_url = sidebar_base_url()
    dialogue_text = st.text_area("Dialogue", value=EXAMPLE_DIALOGUE, height=240)
    patient_id = st.text_input("patient_id (optional)", value="")

    if st.button("Extract SOAP report", type="primary"):
        turns = parse_turns(dialogue_text)
        if not turns:
            st.warning("Enter at least one `speaker: text` line.")
            return
        run_extraction(base_url, turns, patient_id.strip())


if __name__ == "__main__":
    main()
