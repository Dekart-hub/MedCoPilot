"""Streamlit demo UI for the MedCoPilot baseline and the SOAP-correction workflow.

A manual, demo-only interface over the REST API. Two tabs:

* **SOAP extraction** (T16) — paste a doctor-patient dialogue, optionally pin a
  ``patient_id``, and see the extracted ``SoapReport``: the four S/O/A/P
  sections, the ICD code on Assessment claims, per-note confidence, and every
  claim linked back to the dialogue turn it cites.
* **Correction workflow** (stories #8, #10 and #12) — pick a report from a list
  (newest first), open its editable *correction*, and drive the whole lifecycle
  against the API: edit / add / delete notes, re-code the ICD, verify, and
  reopen. The source dialogue is shown alongside, each note shows its origin
  (copied from the original vs doctor-added), and every citation is resolved
  back to its source turn via ``GET /dialogues/{id}``. The same screen carries
  the **LLM correction editor** (story #12): ask the agent for an edit, review
  each proposed operation's before/proposed diff, and accept or reject them one
  by one — pending operations sit in their own prominent section and block
  ``verify`` until decided. The report's changes are shown two ways side by
  side: the doctor's online-quality aggregates (``GET /dialogues/{id}/quality``)
  next to the agent's editor acceptance metric (``GET …/editor/metric``).

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

OPERATION_LABELS = {
    "add_note": "Add note",
    "update_note": "Update note",
    "delete_note": "Delete note",
}

PROPOSAL_STATUS_STYLES = {
    "pending": ("warning", "PENDING — operations still need decisions"),
    "accepted": ("success", "ACCEPTED — every operation was accepted"),
    "rejected": ("error", "REJECTED — every operation was rejected"),
    "mixed": ("info", "MIXED — some operations accepted, some rejected"),
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


def list_reports(base_url: str) -> list[dict[str, str]]:
    """GET ``/reports`` — every report as a summary, already newest-first."""
    response = httpx.get(f"{base_url}/reports", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    result: list[dict[str, str]] = response.json()
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


def get_quality(base_url: str, dialogue_id: str) -> dict[str, Any]:
    """GET dialogue-level online SOAP quality for the verified correction."""
    response = httpx.get(f"{base_url}/dialogues/{dialogue_id}/quality", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def create_editor_proposal(
    base_url: str, report_id: str, user_request: str, patient_id: str
) -> dict[str, Any]:
    """POST an agent edit request; the API records PENDING operations, applies nothing."""
    response = httpx.post(
        f"{base_url}/reports/{report_id}/correction/editor/proposals",
        json={"user_request": user_request, "patient_id": patient_id},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def load_editor_proposal(base_url: str, report_id: str) -> dict[str, Any] | None:
    """GET the active/most-recent proposal; ``None`` when none exists yet (404)."""
    try:
        response = httpx.get(
            f"{base_url}/reports/{report_id}/correction/editor/proposals",
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        st.error(describe_http_error(error))
        return None
    except httpx.RequestError as error:
        st.error(f"Could not reach the API — is it running? ({error})")
        return None
    result: dict[str, Any] = response.json()
    return result


def decide_editor_operation(
    base_url: str, report_id: str, proposal_id: str, operation_id: str, verdict: str
) -> dict[str, Any]:
    """POST accept/reject for one operation and return the reloaded proposal."""
    response = httpx.post(
        f"{base_url}/reports/{report_id}/correction/editor/proposals/"
        f"{proposal_id}/operations/{operation_id}/{verdict}",
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def get_editor_metric(base_url: str, report_id: str) -> dict[str, Any]:
    """GET the operation-level acceptance metric of the report's editor session."""
    response = httpx.get(
        f"{base_url}/reports/{report_id}/correction/editor/metric",
        timeout=REQUEST_TIMEOUT,
    )
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


def call_api[T](action: Callable[..., T], *args: Any, success: str | None = None) -> T | None:
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
        st.info("Switch to the **Correction workflow** tab — this report is pre-selected there.")


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


def render_doctor_change_summary(
    base_url: str, ctx: dict[str, Any], correction: dict[str, Any]
) -> dict[str, Any] | None:
    """Doctor side of the comparison: the online-quality headline, or why it is unavailable."""
    if correction["status"] != "verified":
        st.info("Available after the doctor verifies this correction.")
        return None
    dialogue_id = ctx.get("dialogue_id")
    if not dialogue_id:
        st.warning("Load the source dialogue id to retrieve dialogue-level quality.")
        return None
    quality = call_api(get_quality, base_url, dialogue_id)
    if quality is None:
        return None
    st.metric("Notes added", quality["notes_added"])
    st.metric("Notes removed", quality["notes_removed"])
    st.metric("Changed characters", quality["changed_characters"])
    st.metric("Diagnosis changes", quality["diagnosis_changes"])
    return quality


def render_agent_change_summary(metric: dict[str, Any] | None) -> None:
    """Agent side of the comparison: the LLM-editor acceptance headline."""
    if metric is None:
        return
    st.metric("Accepted", metric["accepted"])
    st.metric("Rejected", metric["rejected"])
    st.metric("Pending", metric["pending"])
    rate = metric["acceptance_rate"]
    st.metric("Acceptance rate", "n/a" if rate is None else f"{rate:.0%}")


def render_quality_note_diffs(quality: dict[str, Any]) -> None:
    """The matched source/corrected note detail behind the doctor-change aggregates."""
    st.caption(
        f"report {quality['report_id']} · correction {quality['correction_id']} · "
        "matched-note character and diagnosis detail (doctor changes)"
    )
    note_diffs = quality["note_diffs"]
    if note_diffs:
        rows = [
            {
                "Source note": diff["source_note_id"],
                "Corrected note": diff["corrected_note_id"],
                "Changed characters": diff["changed_characters"],
                "Diagnosis changed": "Yes" if diff["diagnosis_changed"] else "No",
            }
            for diff in note_diffs
        ]
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.caption("No matched notes — only additions/removals may remain.")
    with st.expander("Raw quality JSON"):
        st.json(quality)


def render_metric_breakdown(metric: dict[str, Any] | None) -> None:
    """The agent acceptance detail sliced by model id and prompt version."""
    if metric is None:
        return
    with st.expander("Agent acceptance detail (by model / prompt version)"):
        breakdown = metric["breakdown"]
        if breakdown:
            rows = [
                {
                    "Model": slice_["model_id"],
                    "Prompt": slice_["prompt_version"],
                    "Proposed": slice_["proposed"],
                    "Accepted": slice_["accepted"],
                    "Rejected": slice_["rejected"],
                    "Pending": slice_["pending"],
                }
                for slice_ in breakdown
            ]
            st.dataframe(rows, hide_index=True, width="stretch")
        else:
            st.caption("No editor operations recorded for this report yet.")
        st.json(metric)


def render_change_comparison(
    base_url: str, ctx: dict[str, Any], correction: dict[str, Any]
) -> None:
    """Render doctor-change quality next to the agent-editor acceptance, side by side."""
    st.divider()
    st.subheader("Report changes — doctor vs agent")
    metric = call_api(get_editor_metric, base_url, ctx["report_id"])
    doctor_col, agent_col = st.columns(2)
    with doctor_col:
        st.markdown("#### Doctor changes")
        st.caption("Online SOAP quality — `GET /dialogues/{id}/quality`.")
        quality = render_doctor_change_summary(base_url, ctx, correction)
    with agent_col:
        st.markdown("#### Agent changes")
        st.caption("LLM-editor acceptance — `GET …/editor/metric`.")
        render_agent_change_summary(metric)
    if quality is not None:
        render_quality_note_diffs(quality)
    render_metric_breakdown(metric)


def render_operation_note(
    ctx: dict[str, Any], note: dict[str, Any] | None, *, show_icd: bool
) -> None:
    """Render one side of an operation's diff: a note's sections, citations, optional ICD."""
    if note is None:
        st.caption("_(none)_")
        return
    sections = note["sections"]
    for section_key in SECTION_TITLES:
        claims = sections.get(section_key, [])
        if not claims:
            continue
        st.markdown(f"**{SECTION_TITLES[section_key]}**")
        for claim in claims:
            st.markdown(f"- {claim['text']}")
            if show_icd and section_key == "assessment":
                render_icd(claim.get("icd"))
            for citation in claim["citations"]:
                render_citation(citation, ctx["turns"])


def render_operation(ctx: dict[str, Any], operation: dict[str, Any]) -> None:
    """Render one operation: its type, target, before/proposed diff and the ICD caveat."""
    st.markdown(f"**{OPERATION_LABELS.get(operation['type'], operation['type'])}**")
    target = operation["target_note_id"]
    if target:
        st.caption(f"target note: `{target[:8]}…`")
    before_col, proposed_col = st.columns(2)
    with before_col:
        st.markdown("_Before_")
        render_operation_note(ctx, operation["before"], show_icd=True)
    with proposed_col:
        st.markdown("_Proposed_")
        render_operation_note(ctx, operation["proposed"], show_icd=False)
    st.caption("The agent never changes ICD codings — an accepted update keeps the note's ICD.")


def _decide_operation(
    base_url: str, ctx: dict[str, Any], proposal: dict[str, Any], op_id: str, verdict: str, msg: str
) -> None:
    """Send one accept/reject decision and rerun so the correction and metric refresh."""
    if call_api(
        decide_editor_operation,
        base_url,
        ctx["report_id"],
        proposal["id"],
        op_id,
        verdict,
        success=msg,
    ):
        st.rerun()


def render_operation_controls(
    base_url: str, ctx: dict[str, Any], proposal: dict[str, Any], operation: dict[str, Any]
) -> None:
    """Per-operation Accept / Reject buttons wired to the editor endpoints."""
    accept_col, reject_col = st.columns(2)
    op_id = operation["id"]
    with accept_col:
        if st.button("Accept", key=f"accept-{op_id}", type="primary"):
            _decide_operation(base_url, ctx, proposal, op_id, "accept", "Accepted — updated.")
    with reject_col:
        if st.button("Reject", key=f"reject-{op_id}"):
            _decide_operation(base_url, ctx, proposal, op_id, "reject", "Rejected — unchanged.")


def render_pending_operations(
    base_url: str,
    ctx: dict[str, Any],
    proposal: dict[str, Any],
    pending: list[dict[str, Any]],
    *,
    decidable: bool,
) -> None:
    """The prominent, distinct section of operations still awaiting a doctor decision."""
    st.markdown("### ⏳ Pending operations — awaiting your decision")
    if not pending:
        st.success("No pending operations — every proposed edit has been decided.")
        return
    st.warning(
        f"{len(pending)} operation(s) still need a decision. "
        "Verify stays blocked until all are decided."
    )
    for operation in pending:
        with st.container(border=True):
            render_operation(ctx, operation)
            if decidable:
                render_operation_controls(base_url, ctx, proposal, operation)


def render_decided_operations(ctx: dict[str, Any], decided: list[dict[str, Any]]) -> None:
    """Operations the doctor has already accepted or rejected, kept out of the pending area."""
    if not decided:
        return
    st.markdown("### Decided operations")
    for operation in decided:
        label = OPERATION_LABELS.get(operation["type"], operation["type"])
        with st.expander(f"{label} · {operation['decision'].upper()}"):
            render_operation(ctx, operation)


def render_status_badge(status: str) -> None:
    """Render the proposal's overall outcome (pending / accepted / rejected / mixed)."""
    level, message = PROPOSAL_STATUS_STYLES.get(status, ("write", status))
    getattr(st, level)(f"Overall status: {message}")


def render_proposal(
    base_url: str, ctx: dict[str, Any], correction: dict[str, Any], proposal: dict[str, Any]
) -> None:
    """Render one proposal: its request, overall status and per-operation review."""
    st.markdown(f"**Agent request:** {proposal['user_request']}")
    render_status_badge(proposal["status"])
    operations = proposal["operations"]
    pending = [op for op in operations if op["decision"] == "pending"]
    decided = [op for op in operations if op["decision"] != "pending"]
    decidable = correction["status"] == "draft"
    render_pending_operations(base_url, ctx, proposal, pending, decidable=decidable)
    render_decided_operations(ctx, decided)
    with st.expander("Raw proposal JSON"):
        st.json(proposal)


def render_propose_form(base_url: str, ctx: dict[str, Any], *, active: bool) -> None:
    """Capture a user request and ask the agent for a proposal (blocked while one is pending)."""
    with st.form("propose-edit"):
        user_request = st.text_area(
            "Ask the agent to edit this correction",
            placeholder="e.g. Add a plan note to schedule a follow-up chest X-ray in two weeks.",
        )
        patient_id = st.text_input("patient_id", value="")
        submitted = st.form_submit_button("Generate proposal", disabled=active)
    if active:
        st.caption("Decide the pending operations below before requesting a new proposal.")
    if submitted:
        if not user_request.strip():
            st.warning("Describe the edit you want the agent to make.")
        elif call_api(
            create_editor_proposal,
            base_url,
            ctx["report_id"],
            user_request.strip(),
            patient_id.strip(),
            success="Proposal generated.",
        ):
            st.rerun()


def render_editor_panel(
    base_url: str,
    ctx: dict[str, Any],
    correction: dict[str, Any],
    proposal: dict[str, Any] | None,
) -> None:
    """The LLM correction editor: request a proposal, then review each operation."""
    st.subheader("LLM correction editor (story #12)")
    st.caption(
        "The agent proposes note edits through the API; the doctor accepts or rejects each one. "
        "ICD codings never travel through a proposal."
    )
    if correction["status"] == "draft":
        render_propose_form(base_url, ctx, active=_has_pending(proposal))
    else:
        st.info("Proposals are generated only on a draft correction — reopen it to use the editor.")
    if proposal is None:
        st.caption("No proposal yet for this report.")
        return
    render_proposal(base_url, ctx, correction, proposal)


def render_verify_controls(base_url: str, ctx: dict[str, Any], *, has_pending: bool) -> None:
    """Draft-only: capture a doctor id and verify — blocked while a proposal has pending ops."""
    st.markdown("#### Verify")
    if has_pending:
        st.warning(
            "Verify is disabled while the agent proposal has pending operations — "
            "decide them in the LLM editor above first. The API also rejects it with a 409."
        )
    doctor_id = st.text_input("Doctor id", key="verify-doctor")
    if st.button("Verify correction", type="primary", disabled=has_pending):
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


def _has_pending(proposal: dict[str, Any] | None) -> bool:
    """True when the current proposal still holds undecided operations."""
    return proposal is not None and proposal["status"] == "pending"


def render_correction(base_url: str, ctx: dict[str, Any], correction: dict[str, Any]) -> None:
    """Render the whole correction and wire every workflow action to the API."""
    editable = correction["status"] == "draft"
    render_correction_header(correction)
    proposal = load_editor_proposal(base_url, ctx["report_id"])
    render_change_comparison(base_url, ctx, correction)
    st.divider()
    for position, note in enumerate(correction["notes"], start=1):
        render_corrected_note(base_url, ctx, note, position, editable=editable)
    st.divider()
    render_editor_panel(base_url, ctx, correction, proposal)
    st.divider()
    if editable:
        if ctx["turns"]:
            render_add_note_form(base_url, ctx)
        else:
            st.info("Load a source dialogue id above to add grounded notes.")
        render_verify_controls(base_url, ctx, has_pending=_has_pending(proposal))
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


def report_option_label(summary: dict[str, str]) -> str:
    """A readable one-line label for a report in the newest-first picker."""
    created = summary["created_at"].replace("T", " ")
    return (
        f"{created} · report {summary['report_id'][:8]}… · dialogue {summary['dialogue_id'][:8]}…"
    )


def default_report_index(summaries: list[dict[str, str]]) -> int:
    """Preselect the just-extracted report if it is listed, else the newest."""
    extracted = st.session_state.get("extracted_report_id")
    for index, summary in enumerate(summaries):
        if summary["report_id"] == extracted:
            return index
    return 0


def render_report_picker(base_url: str) -> None:
    """List reports newest-first and open the chosen one's correction."""
    summaries = call_api(list_reports, base_url)
    if summaries is None:
        return
    if not summaries:
        st.info("No reports yet — extract one in the **SOAP extraction** tab first.")
        return
    summary = st.selectbox(
        "Report (newest first)",
        options=summaries,
        index=default_report_index(summaries),
        format_func=report_option_label,
    )
    if st.button("Open correction", type="primary"):
        load_correction_context(base_url, summary["report_id"], summary["dialogue_id"])


def render_manual_load(base_url: str) -> None:
    """Fallback: open a correction by typing its report and dialogue ids."""
    with st.expander("Load by id (fallback)"):
        report_id = st.text_input("Report id", key="manual-report-id")
        dialogue_id = st.text_input("Source dialogue id", key="manual-dialogue-id")
        if st.button("Load correction", key="manual-load"):
            if not report_id.strip():
                st.warning("Enter a report id.")
            else:
                load_correction_context(base_url, report_id.strip(), dialogue_id.strip())


def render_dialogue(turns: list[dict[str, str]]) -> None:
    """Show the source dialogue in speaking order so citations are readable."""
    with st.expander("Source dialogue", expanded=True):
        if not turns:
            st.caption("_(no turns loaded)_")
            return
        for index, turn in enumerate(turns, start=1):
            st.markdown(f"**Turn {index} · {turn['speaker']}:** {turn['text']}")


def render_correction_tab(base_url: str) -> None:
    """The correction-workflow tab: pick a report, then drive its lifecycle."""
    st.write("Pick a report below to open its correction, then edit → verify → reopen.")
    render_report_picker(base_url)
    render_manual_load(base_url)

    ctx = st.session_state.get("corr_ctx")
    if not ctx:
        return
    correction = call_api(get_correction, base_url, ctx["report_id"])
    if correction is None:
        return
    render_dialogue(ctx["turns"])
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
