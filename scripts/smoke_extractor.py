"""Manual acceptance runner for the SOAP extractor against a live LLM.

Pushes a few prepared dialogues through the real extraction pipeline on a live
vLLM server (MedGemma), validates each resulting ``SoapReport`` against the SOAP
schema invariants, and prints a human-readable pass/fail report.

The pass/fail gate is the schema invariants: S/O/A/P structure is present and
populated, every claim cites a real dialogue turn, and each dialogue yields at
least one note. Verbatim grounding of a quote in its cited turn is a Tier-0
quality signal (see ``soap/score/tier0.py``), so it is shown per claim as
detail but does not flip the verdict.

NOT a pytest test and NOT wired into CI — a QA runs it by hand. It needs a live
vLLM server (task T7) reachable over the network.

Configuration via env (defaults target the local T7 server):
    VLLM_BASE_URL   OpenAI-compatible endpoint, e.g. http://localhost:8001/v1
    MODEL_ID        served model, e.g. google/medgemma-4b-it
    VLLM_API_KEY    key for the endpoint (default ``EMPTY`` — vLLM ignores it)

Run:
    VLLM_BASE_URL=http://localhost:8001/v1 MODEL_ID=google/medgemma-4b-it \
        PYTHONPATH=src uv run python scripts/smoke_extractor.py

Exit code is 0 when every dialogue passes, 1 when any dialogue fails.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import OpenAISettings, Settings  # noqa: E402
from dialogue import Dialogue, DialogueTurnId  # noqa: E402
from infra import build_chat_model  # noqa: E402
from shared.langgraph import LangGraphAgent  # noqa: E402
from shared.prompts import InMemoryPromptStore  # noqa: E402
from soap.extractor import DEFAULT_PROMPTS, LlmSoapExtractor, build_graph  # noqa: E402
from soap.soap import SoapClaim, SoapReport  # noqa: E402

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8001/v1")
MODEL_ID = os.environ.get("MODEL_ID", "google/medgemma-4b-it")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")


@dataclass(frozen=True)
class Fixture:
    """A prepared dialogue plus its patient context for one acceptance case."""

    name: str
    patient: str
    script: str


def _fixtures() -> list[Fixture]:
    return [
        Fixture(
            name="Community-acquired pneumonia",
            patient="M, 45, no chronic conditions, no known allergies",
            script="""
medic Здравствуйте, что вас беспокоит?
person Уже четыре дня высокая температура и кашель с мокротой.
person Ещё тяжело дышать, особенно когда поднимаюсь по лестнице.
medic Давайте послушаю лёгкие. Справа снизу слышны влажные хрипы.
medic Температура сейчас 38 и 6, дыхание учащённое.
person При глубоком вдохе колет в правом боку.
medic Это похоже на внебольничную пневмонию.
medic Назначаю амоксициллин и направляю на рентген грудной клетки.
medic Приходите на контроль через три дня.
""",
        ),
        Fixture(
            name="Arterial hypertension",
            patient="F, 58, overweight, smoker",
            script="""
medic Проходите, на что жалуетесь?
person Последние две недели по утрам болит затылок и шумит в ушах.
person Дома мерил давление — часто около 160 на 100.
medic Хронические болезни есть, что-нибудь принимаете постоянно?
person Постоянно ничего не принимаю, лет пять назад говорили про давление.
medic Измеряю сейчас: 158 на 98, пульс 76, ритмичный.
medic Это артериальная гипертензия второй степени.
medic Назначаю эналаприл 10 мг утром и советую ограничить соль.
medic Ведите дневник давления и приходите через две недели.
""",
        ),
        Fixture(
            name="Acute low back pain",
            patient="M, 34, manual labour job",
            script="""
medic Здравствуйте, что случилось?
person Вчера поднял тяжёлый ящик и резко заболела поясница.
person Боль отдаёт в левую ногу, тяжело наклоняться.
medic Онемение или слабость в ноге есть, тазовые функции не нарушены?
person Онемения нет, слабости тоже, всё в порядке.
medic При осмотре напряжены мышцы поясницы, симптом Ласега слева положительный.
medic Похоже на люмбоишиалгию из-за мышечного спазма.
medic Назначаю ибупрофен и покой на несколько дней, резких нагрузок избегать.
medic Если появится онемение или слабость — сразу на приём.
""",
        ),
    ]


def _settings() -> Settings:
    return Settings(
        openai=OpenAISettings(
            api_key=VLLM_API_KEY,
            model=MODEL_ID,
            base_url=VLLM_BASE_URL,
            temperature=0.0,
        )
    )


def _build_extractor(settings: Settings) -> LlmSoapExtractor:
    """Assemble the production extractor exactly as the DI container does."""
    model = build_chat_model(settings)
    prompts = InMemoryPromptStore(DEFAULT_PROMPTS)
    agent = LangGraphAgent(build_graph(model, prompts))
    return LlmSoapExtractor(agent)


def _build_dialogue(fixture: Fixture) -> Dialogue:
    return Dialogue.from_text(fixture.script, patient_ref=fixture.patient)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _claim_failures(
    label: str,
    claim: SoapClaim,
    turn_ids: set[DialogueTurnId],
) -> list[str]:
    """Schema-invariant violations for a single S/O/A/P claim."""
    failures: list[str] = []
    if not claim.claim.strip():
        failures.append(f"{label}: empty claim text")
    if claim.evidence.turn_id not in turn_ids:
        failures.append(f"{label}: cites a turn absent from the dialogue")
    return failures


def _validate(dialogue: Dialogue, report: SoapReport) -> list[str]:
    """Collect every SOAP-schema violation across the whole report."""
    if not report.soap_notes:
        return ["no SOAP notes were produced (expected at least one)"]
    turn_ids = {turn.id for turn in dialogue.turns}
    failures: list[str] = []
    for position, note in enumerate(report.soap_notes, start=1):
        for section, claim in note.sections():
            failures.extend(
                _claim_failures(f"note {position} / {section}", claim, turn_ids)
            )
    return failures


def _is_grounded(claim: SoapClaim, content_by_id: dict[DialogueTurnId, str]) -> bool:
    content = content_by_id.get(claim.evidence.turn_id)
    if content is None:
        return False
    quote = _normalize(claim.evidence.text)
    return bool(quote) and quote in _normalize(content)


def _grounding_ratio(
    report: SoapReport, content_by_id: dict[DialogueTurnId, str]
) -> tuple[int, int]:
    claims = [claim for note in report.soap_notes for _, claim in note.sections()]
    grounded = sum(_is_grounded(claim, content_by_id) for claim in claims)
    return grounded, len(claims)


def _print_claim(
    section: str,
    claim: SoapClaim,
    index_by_id: dict[DialogueTurnId, int],
    content_by_id: dict[DialogueTurnId, str],
) -> None:
    turn_index = index_by_id.get(claim.evidence.turn_id)
    where = f"turn {turn_index}" if turn_index is not None else "unknown turn"
    mark = "grounded" if _is_grounded(claim, content_by_id) else "NOT grounded"
    print(f"    {section[0].upper()}  {claim.claim}")
    print(f'       cite [{where}, {mark}]: "{claim.evidence.text}"')


def _print_case(
    fixture: Fixture,
    dialogue: Dialogue,
    report: SoapReport,
    failures: list[str],
) -> None:
    index_by_id = {turn.id: i for i, turn in enumerate(dialogue.turns, start=1)}
    content_by_id = {turn.id: turn.content for turn in dialogue.turns}
    print(f"  patient: {fixture.patient}")
    print(f"  dialogue: {len(dialogue.turns)} turns")
    print(f"  notes extracted: {len(report.soap_notes)}")
    for position, note in enumerate(report.soap_notes, start=1):
        print(f"  Note {position}")
        for section, claim in note.sections():
            _print_claim(section, claim, index_by_id, content_by_id)
    grounded, total = _grounding_ratio(report, content_by_id)
    print(f"  grounding (Tier-0 detail): {grounded}/{total} quotes verbatim")
    verdict = "PASS" if not failures else "FAIL"
    print(f"  RESULT: {verdict}")
    for failure in failures:
        print(f"      - {failure}")


async def _run_case(extractor: LlmSoapExtractor, fixture: Fixture) -> bool:
    dialogue = _build_dialogue(fixture)
    report = await extractor.extract(dialogue)
    failures = _validate(dialogue, report)
    _print_case(fixture, dialogue, report, failures)
    return not failures


async def main() -> int:
    print("=" * 64)
    print(" SOAP extractor — live acceptance run")
    print(f" vLLM: {VLLM_BASE_URL}   model: {MODEL_ID}")
    print("=" * 64)

    extractor = _build_extractor(_settings())
    fixtures = _fixtures()

    passed = 0
    for position, fixture in enumerate(fixtures, start=1):
        print(f"\n[{position}/{len(fixtures)}] {fixture.name}")
        passed += await _run_case(extractor, fixture)

    print("\n" + "=" * 64)
    print(f" SUMMARY: {passed}/{len(fixtures)} dialogues passed")
    print("=" * 64)
    return 0 if passed == len(fixtures) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
