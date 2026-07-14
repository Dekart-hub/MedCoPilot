"""Programmatic Alembic access used to auto-migrate the database on startup.

Alembic's command API is synchronous, so callers running inside an event loop
(the app lifespan) must invoke :func:`run_migrations` in a worker thread.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ALEMBIC_INI = _PROJECT_ROOT / "alembic.ini"


def _alembic_config() -> Config:
    return Config(str(_ALEMBIC_INI))


def run_migrations() -> None:
    """Upgrade the database to the latest revision (``alembic upgrade head``)."""
    command.upgrade(_alembic_config(), "head")
