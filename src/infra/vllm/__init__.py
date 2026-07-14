"""vLLM model-serving deployment configuration (see ``deployment.py``)."""

from __future__ import annotations

from infra.vllm.deployment import MEDGEMMA_4B, MEDGEMMA_4B_MODEL_ID, VllmDeployment

__all__ = ["MEDGEMMA_4B", "MEDGEMMA_4B_MODEL_ID", "VllmDeployment"]
