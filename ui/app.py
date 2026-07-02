"""Демо-UI для MedCoPilot на Streamlit.

Тонкий клиент поверх REST API: список диалогов, создание из текста, «проваливание»
в диалог и генерация SOAP-репорта по кнопке. Сознательно простой — только для демо.

Запуск (бэкенд должен быть поднят отдельно):
    uv run --project ui streamlit run ui/app.py
"""

from __future__ import annotations

import os

import requests
import streamlit as st

DEFAULT_API = os.environ.get("MEDCOPILOT_API", "http://localhost:8000")

st.set_page_config(page_title="MedCoPilot — демо", layout="centered")
api_base = st.sidebar.text_input("API base URL", DEFAULT_API).rstrip("/")
st.sidebar.caption("Бэкенд: `uv run uvicorn app.main:app` в каталоге src")


def api_get(path: str):
    return requests.get(f"{api_base}{path}", timeout=120)


def api_post(path: str, payload: dict):
    return requests.post(f"{api_base}{path}", json=payload, timeout=300)


def open_dialogue(dialogue_id: str) -> None:
    st.session_state["dialogue_id"] = dialogue_id
    st.rerun()


# --------------------------------------------------------------------------- #
# Экран списка диалогов + создание из текста.
# --------------------------------------------------------------------------- #


def render_home() -> None:
    st.title("MedCoPilot — демо")

    with st.expander("➕ Новый диалог из текста", expanded=False):
        st.caption("Каждая строка — одна реплика в формате `роль текст`.")
        text = st.text_area(
            "Текст диалога",
            height=160,
            placeholder="person Здравствуйте, третий день болит голова\n"
            "medic Давайте измерим давление...",
        )
        if st.button("Создать диалог", type="primary", disabled=not text.strip()):
            resp = api_post("/api/v1/dialogues/from-text", {"text": text})
            if resp.status_code == 201:
                open_dialogue(resp.json()["id"])
            else:
                st.error(f"Не удалось создать: {resp.status_code} {resp.text}")

    st.subheader("Диалоги")
    try:
        resp = api_get("/api/v1/dialogues")
    except requests.RequestException as exc:
        st.error(f"Бэкенд недоступен по {api_base}: {exc}")
        return

    if resp.status_code != 200:
        st.error(f"Ошибка списка: {resp.status_code} {resp.text}")
        return

    dialogues = resp.json()
    if not dialogues:
        st.info("Диалогов пока нет — создайте из текста выше.")
        return

    for d in dialogues:
        turns = d.get("turns", [])
        preview = turns[0]["content"][:70] if turns else "(пусто)"
        label = f"🗂 {len(turns)} реплик · {preview}…"
        if st.button(label, key=f"open-{d['id']}", use_container_width=True):
            open_dialogue(d["id"])


# --------------------------------------------------------------------------- #
# Экран диалога + генерация репорта.
# --------------------------------------------------------------------------- #


def render_dialogue(dialogue_id: str) -> None:
    if st.button("← К списку"):
        st.session_state.pop("dialogue_id", None)
        st.rerun()

    resp = api_get(f"/api/v1/dialogues/{dialogue_id}")
    if resp.status_code != 200:
        st.error(f"Диалог не найден: {resp.status_code}")
        return

    dialogue = resp.json()
    st.title("Диалог")
    st.caption(dialogue_id)

    for turn in dialogue.get("turns", []):
        with st.chat_message("user" if turn["role"] == "person" else "assistant"):
            st.markdown(f"**{turn['role']}** · {turn['content']}")

    st.divider()
    if st.button("🧾 Сгенерировать SOAP-репорт", type="primary"):
        with st.spinner("Генерация..."):
            report = api_post("/api/v1/reports", {"dialogue_id": dialogue_id})
        if report.status_code == 201:
            st.session_state[f"report-{dialogue_id}"] = report.json()
        else:
            st.error(f"Ошибка генерации: {report.status_code} {report.text}")

    report = st.session_state.get(f"report-{dialogue_id}")
    if report:
        render_report(report)


def render_report(report: dict) -> None:
    st.subheader("SOAP-репорт")
    for i, note in enumerate(report.get("soap_notes", []), 1):
        confidence = note.get("confidence")
        title = f"Запись {i}"
        if confidence is not None:
            title += f"  ·  уверенность {confidence:.0%}"
        with st.expander(title, expanded=True):
            _section("S — Subjective", note["subjective"])
            _section("O — Objective", note["objective"])
            _section("A — Assessment", note["assessment"])
            _selected_coding(note["assessment"])
            _codings(note["assessment"].get("codings", []))
            _section("P — Plan", note["plan"])


def _section(label: str, claim: dict) -> None:
    st.markdown(f"**{label}**")
    st.write(claim["claim"])


def _selected_coding(assessment: dict) -> None:
    selected = assessment.get("selected")
    if not selected:
        return
    st.success(f"**{selected['code']}** — {selected['title']}")
    rationale = assessment.get("rationale")
    if rationale:
        st.caption(rationale)


def _codings(codings: list[dict]) -> None:
    if not codings:
        st.caption("Коды МКБ не подобраны.")
        return
    st.markdown("**Кандидаты (ретрив)**")
    st.dataframe(
        [
            {
                "Код": c["code"],
                "Название": c["title"],
                "Совпадение": c["matched_formulation"],
                "Score": round(c["score"], 2),
                "Версия": c["classifier"].get("version") or "—",
            }
            for c in codings
        ],
        hide_index=True,
        use_container_width=True,
    )


if st.session_state.get("dialogue_id"):
    render_dialogue(st.session_state["dialogue_id"])
else:
    render_home()
