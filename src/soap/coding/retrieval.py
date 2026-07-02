"""Лексический индекс МКБ-10 (Tier 1, BM25) поверх справочников НСИ.

Корпус — формулировки Тома 3 (``S_NAME -> ICD-10``); метаданные кода и
иерархия — из Тома 1 (``MKB_CODE``, ``MKB_NAME``, ``ID_PARENT``, ``ACTUAL``).
72k формулировок умещаются в памяти, поэтому никакого ANN/ElasticSearch:
самописный inverted-index + Okapi BM25 отрабатывают за миллисекунды.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from .coding import DEFAULT_MKB10_REF, ClassifierRef
from .preprocess import normalize

# Тип токенизатора: текст -> мешок токенов-основ. Подменяется на английский
# (``preprocess_en.normalize``) при сборке индекса из ICD-10-CM.
Tokenizer = Callable[[str], list[str]]


@dataclass(frozen=True, slots=True)
class MkbEntry:
    """Запись справочника Тома 1."""

    code: str
    name: str
    parent_id: str | None
    record_id: str
    actual: bool


@dataclass(frozen=True, slots=True)
class RawCandidate:
    """Кандидат до перевода в доменный ``DiagnosisCoding``.

    ``score`` — относительный (0, 1]: BM25 топ-кандидата запроса, нормированный
    на максимум в выдаче. Это не вероятность, а порядковый сигнал.
    """

    code: str
    formulation: str
    score: float


class Bm25Index:
    """Okapi BM25 поверх предтокенизированных документов."""

    def __init__(
        self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75
    ) -> None:
        self._k1 = k1
        self._b = b
        self._doc_len = [len(doc) for doc in documents]
        self._doc_count = len(documents)
        self._avgdl = sum(self._doc_len) / self._doc_count if self._doc_count else 0.0

        # term -> [(doc_id, tf), ...]
        self._postings: dict[str, list[tuple[int, int]]] = {}
        for doc_id, doc in enumerate(documents):
            for term, freq in Counter(doc).items():
                self._postings.setdefault(term, []).append((doc_id, freq))

        self._idf: dict[str, float] = {}
        for term, postings in self._postings.items():
            df = len(postings)
            # +1 под логарифмом не даёт idf уйти в минус для частых термов.
            self._idf[term] = math.log(
                1 + (self._doc_count - df + 0.5) / (df + 0.5)
            )

    def search(self, query_tokens: list[str], top_n: int) -> list[tuple[int, float]]:
        if not self._doc_count or not query_tokens:
            return []
        scores: dict[int, float] = {}
        for term in set(query_tokens):
            postings = self._postings.get(term)
            if postings is None:
                continue
            idf = self._idf[term]
            for doc_id, freq in postings:
                dl = self._doc_len[doc_id]
                denom = freq + self._k1 * (1 - self._b + self._b * dl / self._avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * freq * (self._k1 + 1) / denom
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]


class MkbIndex:
    """Поисковый индекс МКБ-10: матчинг текста -> коды + иерархия для back-off."""

    # Сколько документов тянуть из BM25 до дедупликации по коду.
    _POOL_FACTOR = 10
    _MIN_POOL = 50

    def __init__(
        self,
        *,
        codes: list[str],
        formulations: list[str],
        doc_tokens: list[list[str]],
        entries_by_id: dict[str, MkbEntry],
        entry_by_code: dict[str, MkbEntry],
        classifier: ClassifierRef = DEFAULT_MKB10_REF,
        tokenizer: Tokenizer = normalize,
    ) -> None:
        self._codes = codes
        self._formulations = formulations
        self._bm25 = Bm25Index(doc_tokens)
        self._entries_by_id = entries_by_id
        self._entry_by_code = entry_by_code
        # parent record_id -> дети; для уточнения 4-5-го знака в реранкере.
        self._children_by_id: dict[str, list[MkbEntry]] = {}
        for entry in entries_by_id.values():
            if entry.parent_id:
                self._children_by_id.setdefault(entry.parent_id, []).append(entry)
        self._classifier = classifier
        # Один и тот же токенизатор должен применяться к корпусу и к запросу,
        # иначе основы не совпадут — храним его на индексе.
        self._tokenizer = tokenizer

    @property
    def classifier(self) -> ClassifierRef:
        """Референс на источник кодирования (provenance) для проставления в коды."""
        return self._classifier

    # --- поиск ----------------------------------------------------------- #

    def search(self, text: str, top_n: int = 5) -> list[RawCandidate]:
        pool_size = max(top_n * self._POOL_FACTOR, self._MIN_POOL)
        pool = self._bm25.search(self._tokenizer(text), pool_size)
        if not pool:
            return []

        max_raw = pool[0][1]
        # Дедупликация по коду: pool отсортирован по убыванию, первое
        # вхождение кода — его лучшая формулировка.
        best: dict[str, tuple[float, str]] = {}
        for doc_id, raw in pool:
            code = self._codes[doc_id]
            if code not in best:
                best[code] = (raw, self._formulations[doc_id])

        ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:top_n]
        return [
            RawCandidate(
                code=code,
                formulation=formulation,
                score=raw / max_raw if max_raw > 0 else 0.0,
            )
            for code, (raw, formulation) in ranked
        ]

    # --- метаданные / иерархия ------------------------------------------ #

    def name_of(self, code: str) -> str | None:
        entry = self._entry_by_code.get(code)
        return entry.name if entry else None

    def children_of(self, code: str) -> list[MkbEntry]:
        """Прямые потомки рубрики (уточняющие 4-5-й знак), по коду.

        Пустой список — код листовой либо неизвестен. Вместе с ``parent_chain``
        образует «окрестность» кандидата для LLM-реранка: родители дают
        back-off, дети — уточнение.
        """
        entry = self._entry_by_code.get(code)
        if entry is None:
            return []
        children = self._children_by_id.get(entry.record_id, [])
        return sorted(children, key=lambda e: e.code)

    def parent_chain(self, code: str) -> list[str]:
        """Коды предков от ближайшего родителя к корню (для back-off)."""
        entry = self._entry_by_code.get(code)
        chain: list[str] = []
        seen: set[str] = set()
        while entry and entry.parent_id and entry.parent_id not in seen:
            seen.add(entry.parent_id)
            parent = self._entries_by_id.get(entry.parent_id)
            if parent is None:
                break
            chain.append(parent.code)
            entry = parent
        return chain

    # --- сборка ---------------------------------------------------------- #

    @classmethod
    def from_records(
        cls,
        vol1: list[dict],
        vol3: list[dict],
        classifier: ClassifierRef = DEFAULT_MKB10_REF,
        tokenizer: Tokenizer = normalize,
    ) -> MkbIndex:
        entries_by_id: dict[str, MkbEntry] = {}
        entry_by_code: dict[str, MkbEntry] = {}
        for row in vol1:
            code = row.get("MKB_CODE")
            rec_id = row.get("ID")
            if not code or rec_id is None:
                continue
            entry = MkbEntry(
                code=code,
                name=row.get("MKB_NAME") or "",
                parent_id=str(row["ID_PARENT"]) if row.get("ID_PARENT") else None,
                record_id=str(rec_id),
                actual=str(row.get("ACTUAL")) == "1",
            )
            entries_by_id[str(rec_id)] = entry
            # Для поиска по коду предпочитаем актуальную запись.
            if code not in entry_by_code or entry.actual:
                entry_by_code[code] = entry

        codes: list[str] = []
        formulations: list[str] = []
        doc_tokens: list[list[str]] = []

        def add_doc(code: str, text: str) -> None:
            tokens = tokenizer(text)
            if not tokens:
                return
            codes.append(code)
            formulations.append(text)
            doc_tokens.append(tokens)

        for row in vol3:
            name = row.get("S_NAME")
            code = row.get("ICD-10")
            if name and code:
                add_doc(code, name)

        # Канонические названия Тома 1 — тоже поисковые документы: врач нередко
        # формулирует диагноз дословно как рубрику («Acute pharyngitis,
        # unspecified»), а указатель такие формулировки покрывает не всегда.
        for entry in entry_by_code.values():
            if entry.actual and entry.name:
                add_doc(entry.code, entry.name)

        return cls(
            codes=codes,
            formulations=formulations,
            doc_tokens=doc_tokens,
            entries_by_id=entries_by_id,
            entry_by_code=entry_by_code,
            classifier=classifier,
            tokenizer=tokenizer,
        )

    @classmethod
    def from_jsonl(
        cls,
        vol1_path: str,
        vol3_path: str,
        *,
        base_ref: ClassifierRef = DEFAULT_MKB10_REF,
        tokenizer: Tokenizer = normalize,
    ) -> MkbIndex:
        """Сборка индекса из выгрузки.

        ``base_ref`` задаёт источник кодирования (МКБ-10 НСИ по умолчанию,
        ``DEFAULT_ICD10CM_REF`` для английского); версии справочников
        подтягиваются из meta-файлов рядом с jsonl. ``tokenizer`` — язык
        препроцессинга (``preprocess_en.normalize`` для ICD-10-CM).
        """

        def read(path: str) -> list[dict]:
            with open(path, encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]

        classifier = ClassifierRef(
            system=base_ref.system,
            name=base_ref.name,
            version=_meta_version(vol1_path),
            index_oid=base_ref.index_oid,
            index_version=_meta_version(vol3_path),
        )
        return cls.from_records(
            read(vol1_path), read(vol3_path), classifier, tokenizer
        )


def _meta_version(jsonl_path: str) -> str | None:
    """Версия справочника из meta-файла рядом с выгрузкой (если есть)."""
    meta_path = jsonl_path.removesuffix(".jsonl") + ".meta.json"
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f).get("version")
