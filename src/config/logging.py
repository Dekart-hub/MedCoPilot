"""Structured logging setup.

Configures structlog as the single rendering pipeline for both structlog and
stdlib records (uvicorn's loggers included), so the process emits one coherent
stream: JSON in prod-like environments, human-readable in dev. Routing stdlib
through structlog is what prevents uvicorn's default lines from appearing
unstructured alongside our events.
"""

from __future__ import annotations

import logging
import sys

import structlog
from structlog.types import Processor

from config.settings import Settings, get_settings

_PROD_LIKE_ENVS = frozenset({"prod", "production", "staging"})
_UVICORN_PASSTHROUGH_LOGGERS = ("uvicorn", "uvicorn.error")
_UVICORN_ACCESS_LOGGER = "uvicorn.access"


def configure_logging(settings: Settings | None = None) -> None:
    """Route structlog and stdlib logs through one structured renderer."""
    settings = settings or get_settings()
    shared = _shared_processors()

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            _renderer(settings.app_env),
        ],
    )
    _install_root_handler(formatter)
    _reconcile_uvicorn_loggers()


def _shared_processors() -> list[Processor]:
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]


def _renderer(app_env: str) -> Processor:
    if app_env.lower() in _PROD_LIKE_ENVS:
        return structlog.processors.JSONRenderer()
    return structlog.dev.ConsoleRenderer()


def _install_root_handler(formatter: logging.Formatter) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


def _reconcile_uvicorn_loggers() -> None:
    """Route uvicorn's error/startup logs through root and silence its access log.

    uvicorn decides whether to emit an access line from
    ``uvicorn.access.hasHandlers()`` (see ``h11_impl``), so leaving that logger
    to propagate would revive its default line and duplicate the request event
    our middleware already emits. Detaching it keeps the middleware the single
    source of per-request logging, regardless of the ``--no-access-log`` flag.
    """
    for name in _UVICORN_PASSTHROUGH_LOGGERS:
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True

    access_logger = logging.getLogger(_UVICORN_ACCESS_LOGGER)
    access_logger.handlers = []
    access_logger.propagate = False
