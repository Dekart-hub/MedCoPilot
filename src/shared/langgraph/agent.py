from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph, StateGraph


class LangGraphAgent:
    """Базовая обвязка над графом LangGraph.

    Принимает несобранный ``StateGraph``, компилирует его один раз и хранит
    базовый ``RunnableConfig`` (пока пустой). Метод :meth:`run` запускает граф
    на переданном входе и возвращает итоговое состояние.

    Конкретные агенты (например, экстрактор SOAP) получают этот объект в
    конструктор и работают с ним, ничего не зная о деталях LangGraph.
    """

    def __init__(
        self, graph: StateGraph, config: RunnableConfig | None = None
    ) -> None:
        self._config: RunnableConfig = config or RunnableConfig()
        self._compiled: CompiledStateGraph = graph.compile()

    async def run(
        self, messages: Any, config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        """Запускает граф и возвращает итоговое состояние.

        ``messages`` — вход графа (его начальное состояние). ``config``
        мёржится поверх базового конфигурации агента для конкретного запуска.
        """
        merged: RunnableConfig = {**self._config, **(config or {})}
        return await self._compiled.ainvoke(messages, merged)
