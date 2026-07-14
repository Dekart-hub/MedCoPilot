"""SQLAlchemy ORM tables for the Dialogue aggregate.

These rows live in the infrastructure layer: they register the ``dialogue`` and
``dialogue_turn`` tables on ``Base.metadata`` (so Alembic autogenerate sees
them) and are mapped to/from the pure domain aggregate by the repository
adapter. The domain classes stay free of any ORM concern.
"""

from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from infra.db import Base


class DialogueRow(Base):
    __tablename__ = "dialogue"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    turns: Mapped[list[DialogueTurnRow]] = relationship(
        back_populates="dialogue",
        cascade="all, delete-orphan",
        order_by="DialogueTurnRow.position",
    )


class DialogueTurnRow(Base):
    __tablename__ = "dialogue_turn"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    dialogue_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("dialogue.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column()
    speaker: Mapped[str] = mapped_column(String(255))
    text: Mapped[str] = mapped_column(Text())
    dialogue: Mapped[DialogueRow] = relationship(back_populates="turns")
