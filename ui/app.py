"""Streamlit demo UI for the MedCoPilot baseline and the SOAP-correction workflow.

A manual, demo-only interface over the REST API. Two tabs:

* **SOAP extraction** (T16) — paste a doctor-patient dialogue, optionally pin a
  ``patient_id``, and see the extracted ``SoapReport``: the four S/O/A/P
  sections, the ICD code on Assessment claims, per-note confidence, and every
  claim linked back to the dialogue turn it cites.
* **Correction workflow** (story #8) — open the doctor's editable *correction*
  of a report, then drive the whole lifecycle against the API: edit / add /
  delete notes, re-code the ICD, verify, and reopen. Each note shows its origin
  (copied from the original vs doctor-added) and each citation is resolved back
  to its source turn via ``GET /dialogues/{id}``.

This is **out of scope** for the baseline DoD; it is a thin client over the API
and deliberately kept simple. It talks HTTP only -- it imports nothing from
``src`` so it stays decoupled from the service internals.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
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


def get_dialogue(base_url: str, dialogue_id: str) -> dict[str, Any]:
    """GET ``/dialogues/{id}`` and return the dialogue with its ordered turns."""
    response = httpx.get(f"{base_url}/dialogues/{dialogue_id}", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


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


def start_correction(base_url: str, report_id: str) -> dict[str, Any]:
    """POST to ``/reports/{id}/correction`` to open or resume the draft."""
    response = httpx.post(f"{base_url}/reports/{report_id}/correction", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def get_correction(base_url: str, report_id: str) -> dict[str, Any]:
    """GET ``/reports/{id}/correction`` — the current draft/verified version."""
    response = httpx.get(f"{base_url}/reports/{report_id}/correction", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def update_note(
    base_url: str, report_id: str, note_id: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """PUT a note's replacement sections and return the reloaded correction."""
    response = httpx.put(
        f"{base_url}/reports/{report_id}/correction/notes/{note_id}",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def add_note(base_url: str, report_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """POST a doctor-authored note and return the reloaded correction."""
    response = httpx.post(
        f"{base_url}/reports/{report_id}/correction/notes",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def delete_note(base_url: str, report_id: str, note_id: str) -> dict[str, Any]:
    """DELETE a note and return the reloaded correction."""
    response = httpx.delete(
        f"{base_url}/reports/{report_id}/correction/notes/{note_id}",
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def verify_correction(base_url: str, report_id: str, doctor_id: str) -> dict[str, Any]:
    """POST to ``/verify`` to move the correction draft → verified."""
    response = httpx.post(
        f"{base_url}/reports/{report_id}/correction/verify",
        json={"doctor_id": doctor_id},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def reopen_correction(base_url: str, report_id: str) -> dict[str, Any]:
    """POST to ``/reopen`` to move the correction verified → draft."""
    response = httpx.post(
        f"{base_url}/reports/{report_id}/correction/reopen", timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def describe_http_error(error: httpx.HTTPStatusError) -> str:
    """Turn a non-2xx response into a readable one-line ``code: detail`` message."""
    try:
        body = error.response.json()
    except ValueError:
        return f"API returned {error.response.status_code}: {error.response.text}"
    detail = body.get("detail", error.response.text)
    code = body.get("code")
    prefix = f"{code}: " if code else ""
    return f"API returned {error.response.status_code}: {prefix}{detail}"


def call_api(
    action: Callable[..., dict[str, Any]], *args: Any, success: str | None = None
) -> dict[str, Any] | None:
    """Run an API call, surfacing errors as readable messages; ``None`` on failure."""
    try:
        result = action(*args)
    except httpx.HTTPStatusError as error:
        st.error(describe_http_error(error))
        return None
    except httpx.RequestError as error:
        st.error(f"Could not reach the API — is it running? ({error})")
        return None
    if success:
        st.success(success)
    return result


def turn_label(index: int, turn: dict[str, str]) -> str:
    """A short, human-readable label for a dialogue turn used in pickers."""
    text = turn["text"]
    snippet = text if len(text) <= 50 else f"{text[:50]}…"
    return f"Turn {index} · {turn['speaker']}: {snippet}"


def turn_choices(
    turns: list[dict[str, str]],
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Return ``(labels, label→turn_id, turn_id→label)`` for citation pickers."""
    labels: list[str] = []
    label_to_id: dict[str, str] = {}
    id_to_label: dict[str, str] = {}
    for index, turn in enumerate(turns, start=1):
        label = turn_label(index, turn)
        labels.append(label)
        label_to_id[label] = turn["id"]
        id_to_label[turn["id"]] = label
    return labels, label_to_id, id_to_label


def render_citation(citation: dict[str, Any], turns: list[dict[str, str]]) -> None:
    """Render one claim→turn link, resolving the turn id back to its source turn."""
    turn_id = citation["turn_id"]
    match = next(
        ((index, turn) for index, turn in enumerate(turns, start=1) if turn["id"] == turn_id),
        None,
    )
    if match is not None:
        index, turn = match
        st.markdown(f"↳ **Turn {index}** — _{turn['speaker']}_: {turn['text']}")
    else:
        st.markdown(f"↳ source turn `{turn_id[:8]}…`")
    if citation.get("quote"):
        st.caption(f'cited: "{citation["quote"]}"')


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


def run_extraction(base_url: str, turns_input: list[dict[str, str]], patient_id: str) -> None:
    """Create the dialogue, extract its report, and render it (or an error)."""
    try:
        with st.spinner("Creating dialogue and extracting report…"):
            dialogue_id = create_dialogue(base_url, turns_input)
            turns = get_dialogue(base_url, dialogue_id)["turns"]
            report = extract_report(base_url, dialogue_id, patient_id)
    except httpx.HTTPStatusError as error:
        st.error(describe_http_error(error))
    except httpx.RequestError as error:
        st.error(f"Could not reach the API at {base_url} — is it running? ({error})")
    else:
        st.session_state["extracted_report_id"] = report["id"]
        st.session_state["extracted_dialogue_id"] = dialogue_id
        st.success(f"Extracted report for dialogue {dialogue_id}")
        render_report(report, turns)
        st.info("Switch to the **Correction workflow** tab — the ids are pre-filled there.")


def icd_inputs(key: str, icd: dict[str, str] | None) -> dict[str, str] | None:
    """Three ICD fields; return the coding only when all three are filled."""
    code = st.text_input("ICD code", value=(icd or {}).get("code", ""), key=f"{key}-icd-code")
    name = st.text_input("ICD name", value=(icd or {}).get("name", ""), key=f"{key}-icd-name")
    url = st.text_input(
        "ICD classifier URL", value=(icd or {}).get("classifier_url", ""), key=f"{key}-icd-url"
    )
    if code.strip() and name.strip() and url.strip():
        return {"code": code.strip(), "name": name.strip(), "classifier_url": url.strip()}
    return None


def claim_input_row(
    section_key: str,
    key: str,
    labels: list[str],
    label_to_id: dict[str, str],
    id_to_label: dict[str, str],
    *,
    claim: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    """Render one claim's inputs; return ``(payload_or_None, is_valid)``.

    An empty text means "omit this claim" (valid). A non-empty claim that cites
    no turn is invalid — every claim must be grounded in the source dialogue.
    """
    text = st.text_input("Claim text", value=(claim or {}).get("text", ""), key=f"{key}-text")
    cited_ids = [citation["turn_id"] for citation in (claim or {}).get("citations", [])]
    default = [id_to_label[turn_id] for turn_id in cited_ids if turn_id in id_to_label]
    chosen = st.multiselect("Cites turns", options=labels, default=default, key=f"{key}-cites")
    icd = icd_inputs(key, (claim or {}).get("icd")) if section_key == "assessment" else None
    if not text.strip():
        return None, True
    if not chosen:
        return None, False
    payload: dict[str, Any] = {
        "text": text.strip(),
        "citations": [{"turn_id": label_to_id[label]} for label in chosen],
    }
    if section_key == "assessment" and icd is not None:
        payload["icd"] = icd
    return payload, True


def collect_note_sections(
    note: dict[str, Any] | None,
    key_prefix: str,
    labels: list[str],
    label_to_id: dict[str, str],
    id_to_label: dict[str, str],
) -> tuple[dict[str, list[dict[str, Any]]], bool]:
    """Render an editable claim row per section (one blank row when adding)."""
    sections: dict[str, list[dict[str, Any]]] = {}
    valid = True
    for section_key in SECTION_TITLES:
        st.markdown(f"**{SECTION_TITLES[section_key]}**")
        existing = (note or {}).get("sections", {}).get(section_key, []) or [None]
        claims: list[dict[str, Any]] = []
        for index, claim in enumerate(existing):
            key = f"{key_prefix}-{section_key}-{index}"
            payload, ok = claim_input_row(
                section_key, key, labels, label_to_id, id_to_label, claim=claim
            )
            valid = valid and ok
            if payload is not None:
                claims.append(payload)
        sections[section_key] = claims
    return sections, valid


def render_note_editor(base_url: str, ctx: dict[str, Any], note: dict[str, Any]) -> None:
    """An expander form to replace a note's sections, citations and ICD."""
    labels, label_to_id, id_to_label = turn_choices(ctx["turns"])
    with st.expander("Edit note"):
        st.caption("Clear a claim's text to drop it; every kept claim must cite a turn.")
        with st.form(f"edit-{note['id']}"):
            sections, valid = collect_note_sections(
                note, f"edit-{note['id']}", labels, label_to_id, id_to_label
            )
            submitted = st.form_submit_button("Save changes")
        if submitted:
            if not valid:
                st.warning("Every kept claim must cite at least one turn.")
            elif call_api(update_note, base_url, ctx["report_id"], note["id"], sections):
                st.rerun()


def render_add_note_form(base_url: str, ctx: dict[str, Any]) -> None:
    """An expander form to add a doctor-authored note (``source_note_id: null``)."""
    labels, label_to_id, id_to_label = turn_choices(ctx["turns"])
    with st.expander("Add a doctor-authored note"):
        st.caption("Fill any sections; each filled claim must cite at least one turn.")
        with st.form("add-note"):
            sections, valid = collect_note_sections(None, "add", labels, label_to_id, id_to_label)
            submitted = st.form_submit_button("Add note")
        if submitted:
            if not any(sections.values()):
                st.warning("Enter at least one claim.")
            elif not valid:
                st.warning("Every filled claim must cite at least one turn.")
            elif call_api(add_note, base_url, ctx["report_id"], sections):
                st.rerun()


def render_corrected_note(
    base_url: str, ctx: dict[str, Any], note: dict[str, Any], position: int, *, editable: bool
) -> None:
    """Render one corrected note: origin, sections, and (in draft) its edit tools."""
    origin = "copied from original" if note["source_note_id"] else "doctor-added"
    st.markdown(f"### Note {position} · _{origin}_")
    for section_key in SECTION_TITLES:
        render_section(section_key, note["sections"].get(section_key, []), ctx["turns"])
    if not editable:
        return
    if ctx["turns"]:
        render_note_editor(base_url, ctx, note)
    else:
        st.caption("Load a source dialogue id to edit this note's citations.")
    if st.button("Delete note", key=f"delete-{note['id']}") and call_api(
        delete_note, base_url, ctx["report_id"], note["id"]
    ):
        st.rerun()


def render_correction_header(correction: dict[str, Any]) -> None:
    """Show the correction's status and, when verified, its doctor stamp."""
    st.subheader(f"Correction — {correction['status'].upper()}")
    st.caption(f"correction id: {correction['id']} · {len(correction['notes'])} note(s)")
    if correction["status"] == "verified":
        st.success(f"Verified by **{correction['verified_by']}** at {correction['verified_at']}")
    else:
        st.info("Draft — editable. Verify it below once the notes are correct.")


def render_verify_controls(base_url: str, ctx: dict[str, Any]) -> None:
    """Draft-only: capture a doctor id and verify the correction."""
    st.markdown("#### Verify")
    doctor_id = st.text_input("Doctor id", key="verify-doctor")
    if st.button("Verify correction", type="primary"):
        if not doctor_id.strip():
            st.warning("Enter a doctor id to verify.")
        elif call_api(
            verify_correction, base_url, ctx["report_id"], doctor_id.strip(), success="Verified."
        ):
            st.rerun()


def render_locked_controls(base_url: str, ctx: dict[str, Any]) -> None:
    """Verified-only: reopen for editing, or demonstrate the 409 edit lock."""
    st.warning("This correction is verified and locked. Reopen it to edit again.")
    reopen_col, attempt_col = st.columns(2)
    with reopen_col:
        if st.button("Reopen for editing") and call_api(
            reopen_correction, base_url, ctx["report_id"], success="Reopened."
        ):
            st.rerun()
    with attempt_col:
        if st.button("Attempt an edit (should be blocked)"):
            call_api(add_note, base_url, ctx["report_id"], {})


def render_correction(base_url: str, ctx: dict[str, Any], correction: dict[str, Any]) -> None:
    """Render the whole correction and wire every workflow action to the API."""
    editable = correction["status"] == "draft"
    render_correction_header(correction)
    for position, note in enumerate(correction["notes"], start=1):
        render_corrected_note(base_url, ctx, note, position, editable=editable)
    st.divider()
    if editable:
        if ctx["turns"]:
            render_add_note_form(base_url, ctx)
        else:
            st.info("Load a source dialogue id above to add grounded notes.")
        render_verify_controls(base_url, ctx)
    else:
        render_locked_controls(base_url, ctx)
    with st.expander("Raw correction JSON"):
        st.json(correction)


def load_correction_context(base_url: str, report_id: str, dialogue_id: str) -> None:
    """Fetch the source dialogue turns and open the correction; store the context."""
    turns: list[dict[str, str]] = []
    if dialogue_id:
        dialogue = call_api(get_dialogue, base_url, dialogue_id)
        if dialogue is None:
            return
        turns = dialogue["turns"]
    if call_api(start_correction, base_url, report_id, success="Correction ready.") is None:
        return
    st.session_state["corr_ctx"] = {
        "report_id": report_id,
        "dialogue_id": dialogue_id,
        "turns": turns,
    }


def render_correction_tab(base_url: str) -> None:
    """The correction-workflow tab: pick a report, then drive its lifecycle."""
    st.write("Open a report's correction, then edit → verify → reopen against the API.")
    report_id = st.text_input("Report id", value=st.session_state.get("extracted_report_id", ""))
    dialogue_id = st.text_input(
        "Source dialogue id (resolves citations to turn text)",
        value=st.session_state.get("extracted_dialogue_id", ""),
    )
    if st.button("Load correction", type="primary"):
        if not report_id.strip():
            st.warning("Enter a report id.")
        else:
            load_correction_context(base_url, report_id.strip(), dialogue_id.strip())

    ctx = st.session_state.get("corr_ctx")
    if not ctx:
        return
    correction = call_api(get_correction, base_url, ctx["report_id"])
    if correction is None:
        return
    render_correction(base_url, ctx, correction)


def render_extraction_tab(base_url: str) -> None:
    """The extraction tab (T16): paste a dialogue and extract its SOAP report."""
    st.write("Paste a dialogue as `speaker: text` lines, then extract its SOAP report.")
    dialogue_text = st.text_area("Dialogue", value=EXAMPLE_DIALOGUE, height=240)
    patient_id = st.text_input("patient_id (optional)", value="")
    if st.button("Extract SOAP report", type="primary"):
        turns = parse_turns(dialogue_text)
        if not turns:
            st.warning("Enter at least one `speaker: text` line.")
            return
        run_extraction(base_url, turns, patient_id.strip())


def sidebar_base_url() -> str:
    """Render the sidebar and return the configured API base URL."""
    with st.sidebar:
        st.header("Settings")
        base_url = st.text_input("API base URL", value=DEFAULT_API_URL)
        st.caption("Start the baseline API first: `make run` (defaults to :8000).")
    return base_url.rstrip("/")


def main() -> None:
    """Entry point: two tabs over the API — SOAP extraction and correction."""
    st.set_page_config(page_title="MedCoPilot demo", page_icon="🩺")
    st.title("MedCoPilot — SOAP demo")

    base_url = sidebar_base_url()
    extraction_tab, correction_tab = st.tabs(["SOAP extraction", "Correction workflow"])
    with extraction_tab:
        render_extraction_tab(base_url)
    with correction_tab:
        render_correction_tab(base_url)


if __name__ == "__main__":
    main()
