"""API tests for reading a dialogue back, without Postgres.

The database is replaced through FastAPI dependency overrides: an in-memory
dialogue repository stands in for the table and a stub session accepts the
commit. That keeps the test hermetic while exercising the real ``GET
/dialogues/{id}`` route — the ordered turns it returns and its 404 path.
"""

from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from app.dependencies import get_dialogue_repository
from app.main import create_app
from dialogue.dialogue import Dialogue, DialogueId
from dialogue.repository import DialogueRepository
from infra.db import get_session


class FakeSession:
    async def flush(self) -> None: ...

    async def rollback(self) -> None: ...

    async def commit(self) -> None: ...


class InMemoryDialogueRepository(DialogueRepository):
    def __init__(self) -> None:
        self._store: dict[object, Dialogue] = {}

    async def save(self, dialogue: Dialogue) -> None:
        self._store[dialogue.id.value] = dialogue

    async def get(self, dialogue_id: DialogueId) -> Dialogue | None:
        return self._store.get(dialogue_id.value)


def _client() -> TestClient:
    session = FakeSession()
    dialogues = InMemoryDialogueRepository()
    app = create_app()
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_dialogue_repository] = lambda: dialogues
    return TestClient(app)


def test_get_returns_the_dialogue_turns_in_order() -> None:
    client = _client()
    created = client.post(
        "/dialogues",
        json={
            "turns": [
                {"speaker": "patient", "text": "I've had a headache for three days."},
                {"speaker": "doctor", "text": "Blood pressure is 140 over 90."},
            ]
        },
    )
    dialogue_id = created.json()["id"]

    response = client.get(f"/dialogues/{dialogue_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == dialogue_id
    speakers = [turn["speaker"] for turn in body["turns"]]
    texts = [turn["text"] for turn in body["turns"]]
    assert speakers == ["patient", "doctor"]
    assert texts == [
        "I've had a headache for three days.",
        "Blood pressure is 140 over 90.",
    ]
    assert all(turn["id"] for turn in body["turns"])


def test_get_unknown_dialogue_is_404() -> None:
    client = _client()

    response = client.get(f"/dialogues/{uuid4()}")

    assert response.status_code == 404
