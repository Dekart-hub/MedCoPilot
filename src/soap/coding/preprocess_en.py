"""Нормализация текста для лексического матчинга диагнозов (английский, Tier 1).

Английский аналог :mod:`preprocess`: та же контрактная функция ``normalize`` —
текст -> мешок токенов-основ, инвариантный к порядку слов и словоформам.
Применяется и к корпусу (Alphabetic Index ICD-10-CM, офлайн), и к тексту
ассессмента (онлайн). Подключается в индекс через параметр ``tokenizer``.

Стемминг — алгоритм Snowball (English) из пакета ``snowballstemmer``: сводит
``fractured``/``fracture`` к общей основе ``fractur``, чего грубое снятие
суффиксов не делало. Аббревиатуры и стоп-слова остаются нашими — это доменный
слой поверх стеммера.

ВАЖНО про стоп-слова: ``with``/``without``/``no``/``not`` НЕ выкидываем — в
ICD-10-CM именно они различают 4-5-й знак кода («with hyperosmolarity»,
«without coma»), а это ровно то, что дизамбигуирует диагноз.
"""

from __future__ import annotations

import re

import snowballstemmer

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Раскрытие частых клинических аббревиатур. Ключи — «сырые» токены до стемминга.
# Список намеренно короткий и бесспорный; расширяется по мере анализа данных.
ABBREVIATIONS: dict[str, str] = {
    "t2dm": "type 2 diabetes mellitus",
    "t1dm": "type 1 diabetes mellitus",
    "dm": "diabetes mellitus",
    "copd": "chronic obstructive pulmonary disease",
    "htn": "hypertension",
    "chf": "congestive heart failure",
    "cad": "coronary artery disease",
    "mi": "myocardial infarction",
    "ckd": "chronic kidney disease",
    "uti": "urinary tract infection",
    "gerd": "gastroesophageal reflux disease",
    "cva": "cerebrovascular accident",
    "afib": "atrial fibrillation",
    "uri": "upper respiratory infection",
}

# Стоп-слова: только служебные. Клинически значимые with/without/no/not оставлены
# сознательно (см. докстринг модуля). Также выкидываем ICD-жаргон NEC/NOS — врач
# им не говорит, а в корпусе он только шумит.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "to", "in", "on", "for",
        "by", "at", "as", "from", "due",
        "nec", "nos",
    }
)

# Stateless, потокобезопасен на чтение — один экземпляр на процесс.
_STEMMER = snowballstemmer.stemmer("english")


def _stem(token: str) -> str:
    return _STEMMER.stemWord(token)


def normalize(text: str) -> list[str]:
    """Текст -> список токенов-основ для лексического матчинга."""
    expanded: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        expansion = ABBREVIATIONS.get(token)
        if expansion is not None:
            expanded.extend(_TOKEN_RE.findall(expansion))
        else:
            expanded.append(token)

    return [_stem(t) for t in expanded if t not in STOPWORDS]
