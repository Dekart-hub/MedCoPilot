import structlog
from fastapi.testclient import TestClient

from app.main import app


def test_health_returns_ok() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_request_emits_structured_log_event() -> None:
    with structlog.testing.capture_logs() as events:
        TestClient(app).get("/health")

    request_logs = [event for event in events if event["event"] == "http_request"]
    assert len(request_logs) == 1

    log = request_logs[0]
    assert log["request_id"]
    assert log["method"] == "GET"
    assert log["path"] == "/health"
    assert log["status_code"] == 200
    assert isinstance(log["duration_ms"], float)
