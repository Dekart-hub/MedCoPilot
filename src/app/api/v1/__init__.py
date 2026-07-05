from fastapi import APIRouter

from .dialogues import router as dialogues_router
from .reports import router as reports_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(dialogues_router)
api_router.include_router(reports_router)

__all__ = ["api_router"]
