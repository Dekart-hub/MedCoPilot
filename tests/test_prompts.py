from __future__ import annotations

import asyncio

import pytest

from shared.prompts import InMemoryPromptStore, render


def test_render_substitutes_with_and_without_spaces():
    assert render("Hi {{ name }} / {{age}}", {"name": "Bob", "age": 5}) == "Hi Bob / 5"


def test_render_missing_value_raises():
    with pytest.raises(KeyError):
        render("Hi {{ name }}", {})


def test_render_without_placeholders_is_identity():
    assert render("plain text", {}) == "plain text"


def test_store_returns_rendered_prompt():
    store = InMemoryPromptStore({"greet": "Hello {{ who }}"})
    assert asyncio.run(store.get("greet", who="world")) == "Hello world"


def test_store_unknown_key_raises():
    store = InMemoryPromptStore({"greet": "Hello {{ who }}"})
    with pytest.raises(KeyError):
        asyncio.run(store.get("missing", who="world"))
