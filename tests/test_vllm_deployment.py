from __future__ import annotations

from infra.vllm import MEDGEMMA_4B, MEDGEMMA_4B_MODEL_ID, VllmDeployment


def test_medgemma_default_command_args() -> None:
    args = MEDGEMMA_4B.command_args()

    assert args[:2] == ["--model", MEDGEMMA_4B_MODEL_ID]
    # gemma3 must run in bfloat16; float16 is numerically unstable for it.
    assert args[args.index("--dtype") + 1] == "bfloat16"
    # Raised above vLLM's default so the NLI scorer can read yes/no logits.
    assert args[args.index("--max-logprobs") + 1] == "20"


def test_api_key_is_omitted_when_unset() -> None:
    assert "--api-key" not in VllmDeployment().command_args()


def test_api_key_is_forwarded_when_set() -> None:
    args = VllmDeployment(api_key="secret").command_args()

    assert args[args.index("--api-key") + 1] == "secret"


def test_base_url_targets_the_service_host_and_container_port() -> None:
    assert MEDGEMMA_4B.base_url("vllm") == "http://vllm:8000/v1"
