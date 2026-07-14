from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "src" / "infra" / "vllm" / "serve_nli.sh"
)


def _run_script(tmp_path: Path, api_key: str | None) -> list[str]:
    fake_vllm = tmp_path / "vllm"
    fake_vllm.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@"\n', encoding="utf-8")
    fake_vllm.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{tmp_path}:{env['PATH']}",
            "NLI_MODEL_ID": "google/medgemma-4b-it",
            "VLLM_DTYPE": "bfloat16",
        }
    )
    if api_key is None:
        env.pop("VLLM_API_KEY", None)
    else:
        env["VLLM_API_KEY"] = api_key
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout.splitlines()


@pytest.mark.parametrize("api_key", [None, ""])
def test_api_key_can_be_omitted(tmp_path: Path, api_key: str | None):
    args = _run_script(tmp_path, api_key)
    assert args[:2] == ["serve", "google/medgemma-4b-it"]
    assert "--api-key" not in args


def test_api_key_is_forwarded_when_set(tmp_path: Path):
    args = _run_script(tmp_path, "test-key")
    key_index = args.index("--api-key")
    assert args[key_index + 1] == "test-key"
