"""Ручной smoke-тест NliGroundingScorer на реальной модели.

Не входит в pytest: скачивает веса модели при первом запуске (кешируются в
~/.cache/huggingface) и требует transformers/torch (``uv sync --extra nli``).

Запуск с моделью по умолчанию (мультиязычная mDeBERTa, EN+RU):
    uv run python scripts/smoke_nli_scorer.py

Запуск с другой моделью (например, русскоязычной RuBERT-NLI):
    uv run python scripts/smoke_nli_scorer.py --model cointegrated/rubert-base-cased-nli-threeway

Русскоязычные модели типа RuBERT не рассчитаны на английский вход — EN-кейсы
в этом случае ожидаемо могут дать мусорный результат, это не баг скорера.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dialogue import Dialogue, DialogueTurn  # noqa: E402
from shared.value_objects import Id  # noqa: E402
from soap.soap import SoapClaim, SoapEvidence, SoapNote  # noqa: E402
from soap.score.scorer import NliGroundingScorer  # noqa: E402


def _turn(content: str) -> DialogueTurn:
    return DialogueTurn(
        id=Id.new(), role="doctor", content=content, timestamp=datetime.now(timezone.utc)
    )


def _claim(text: str, turn: DialogueTurn) -> SoapClaim:
    return SoapClaim(id=Id.new(), claim=text, evidence=SoapEvidence(text=text, turn_id=turn.id))


def _one_claim_note(claim: SoapClaim) -> SoapNote:
    # Для смока держим все 4 секции одинаковыми, чтобы score ноты = score секции.
    return SoapNote(id=Id.new(), subjective=claim, objective=claim, assessment=claim, plan=claim)


CASES: list[tuple[str, str, str]] = [
    (
        "EN entailment (перефразировка)",
        "Patient reports severe chest pain radiating to the left arm since this morning.",
        "The patient has chest pain.",
    ),
    (
        "EN contradiction (отрицание — галлюцинация)",
        "Patient denies any chest pain or shortness of breath.",
        "The patient has chest pain.",
    ),
    (
        "RU entailment (перефразировка)",
        "Жалуется на сильную головную боль в затылке с сегодняшнего утра.",
        "У пациента головная боль.",
    ),
    (
        "RU contradiction (отрицание — галлюцинация)",
        "Головную боль отрицает, тошноты и рвоты не было.",
        "У пациента головная боль.",
    ),
    (
        "RU neutral (не связано)",
        "Жалуется на сильную головную боль в затылке с сегодняшнего утра.",
        "У пациента температура 38.5.",
    ),
    (
        "RU contradiction (другая локализация — тонкая подмена)",
        "Боль в правом колене при ходьбе, отёка нет.",
        "У пациента болит левое колено.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=NliGroundingScorer._DEFAULT_MODEL,
        help="Имя модели на Hugging Face Hub (AutoModelForSequenceClassification)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    print(f"Загружаю {args.model} (может занять минуту)...")
    scorer = NliGroundingScorer(model_name=args.model)
    print("Модель загружена.\n")

    for label, source_text, claim_text in CASES:
        turn = _turn(source_text)
        claim = _claim(claim_text, turn)
        dialogue = Dialogue(id=Id.new(), turns=[turn], created_at=datetime.now(timezone.utc))
        note = _one_claim_note(claim)

        result = await scorer.score(dialogue, note)
        print(f"[{label}]")
        print(f"  реплика: {source_text}")
        print(f"  claim:   {claim_text}")
        print(f"  score:   {result.score.score:.3f}\n")


if __name__ == "__main__":
    asyncio.run(main())
