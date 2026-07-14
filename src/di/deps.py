from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status

from dialogue import CreateDialogue, CreateDialogueFromText, DialogueRepository
from ehr import ReportRepository, ReportWorkflow
from soap import ExtractScoredSoap

from .container import Container


def get_container(request: Request) -> Container:
    """Достаёт собранный контейнер из состояния приложения.

    Пока зависимости не подняты (см. lifespan), отвечаем 503.
    """
    container: Container | None = getattr(request.app.state, "container", None)
    if container is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service is not ready"
        )
    return container


def get_dialogue_repository(
    container: Container = Depends(get_container),
) -> DialogueRepository:
    return container.dialogue_repository


def get_create_dialogue(
    container: Container = Depends(get_container),
) -> CreateDialogue:
    return container.create_dialogue


def get_create_dialogue_from_text(
    container: Container = Depends(get_container),
) -> CreateDialogueFromText:
    return container.create_dialogue_from_text


def get_extract_scored_soap(
    container: Container = Depends(get_container),
) -> ExtractScoredSoap:
    return container.extract_scored_soap


def get_report_repository(
    container: Container = Depends(get_container),
) -> ReportRepository:
    return container.report_repository


def get_report_workflow(
    container: Container = Depends(get_container),
) -> ReportWorkflow:
    return container.report_workflow
