"""Deployment configuration for the MedGemma model served by vLLM.

MedGemma 4B runs as a standalone vLLM service exposing an OpenAI-compatible API
(the ``vllm`` service in ``docker-compose.yml``). It serves both the SOAP
extractor (T8) and the NLI groundedness scorer (#13); this module is the single
reusable source of truth for how that server is launched and where clients reach
it, so the compose command and downstream clients cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

# MedGemma 4B instruction-tuned checkpoint on the Hugging Face Hub. The model is
# gated: its license must be accepted and an HF token supplied to the server.
MEDGEMMA_4B_MODEL_ID = "google/medgemma-4b-it"


@dataclass(frozen=True, slots=True)
class VllmDeployment:
    """How one model is served by vLLM and reached by clients.

    ``command_args`` assembles the arguments for the vLLM OpenAI server (the
    ``vllm/vllm-openai`` image entrypoint); ``base_url`` resolves the
    OpenAI-compatible endpoint clients POST to.

    Defaults target MedGemma 4B. gemma3 requires ``bfloat16`` on a GPU of compute
    capability >= 8.0 (Ampere or newer): float16 is numerically unstable for it
    and older cards cannot run bf16. ``max_logprobs`` is raised above vLLM's
    default so the NLI scorer (#13) can read the yes/no token logits it needs.
    """

    model_id: str = MEDGEMMA_4B_MODEL_ID
    host: str = "0.0.0.0"
    port: int = 8000
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    max_logprobs: int = 20
    api_key: str | None = None

    def command_args(self) -> list[str]:
        """Arguments passed to the vLLM OpenAI server for this deployment."""
        args = [
            "--model",
            self.model_id,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--dtype",
            self.dtype,
            "--max-model-len",
            str(self.max_model_len),
            "--max-logprobs",
            str(self.max_logprobs),
        ]
        if self.api_key:
            args += ["--api-key", self.api_key]
        return args

    def base_url(self, host: str = "localhost") -> str:
        """OpenAI-compatible base URL clients use to reach this server.

        Inside the compose network ``host`` is the service name (``vllm``); the
        port is always the container port, never a host publish mapping.
        """
        return f"http://{host}:{self.port}/v1"


# Default deployment for the SOAP extractor and the NLI scorer.
MEDGEMMA_4B = VllmDeployment()
