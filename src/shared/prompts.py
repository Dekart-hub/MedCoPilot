from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

_PLACEHOLDER_RE = re.compile(r"{{\s*(\w+)\s*}}")


def render(template: str, values: dict[str, Any]) -> str:
    """Классическая подстановка ``{{ key }}`` → значение.

    Поддерживает пробелы внутри скобок (``{{key}}`` и ``{{ key }}``).
    Бросает :class:`KeyError`, если в шаблоне есть плейсхолдер без значения —
    лучше упасть на сборке промта, чем уйти в LLM с дыркой в тексте.
    """

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            raise KeyError(f"No value for placeholder {{{{ {name} }}}}")
        return str(values[name])

    return _PLACEHOLDER_RE.sub(replace, template)


class PromptStore(ABC):
    """Хранилище промтов: по ключу отдаёт шаблон с подставленными значениями.

    Интерфейс асинхронный: за ним может стоять не только память, но и удалённый
    источник (например, LangFuse) с сетевым вызовом.
    """

    @abstractmethod
    async def get(self, key: str, /, **values: Any) -> str:
        raise NotImplementedError


class InMemoryPromptStore(PromptStore):
    """Промты из словаря в памяти (ключ → шаблон)."""

    def __init__(self, templates: dict[str, str]) -> None:
        self._templates = dict(templates)

    async def get(self, key: str, /, **values: Any) -> str:
        try:
            template = self._templates[key]
        except KeyError:
            raise KeyError(f"Unknown prompt key: {key!r}") from None
        return render(template, values)
