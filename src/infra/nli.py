"""Settings-driven construction of the NLI-backed confidence scorer.

The infra edge for [#7/FR-4]: it turns :class:`Settings` into a ready
:class:`~soap.nli_scorer.NliConfidenceScorer` wrapping a
:class:`~nli.VllmNliScorer`. The model-specific bits (the validated MedGemma
yes/no chat-template prompt, the tokenizer id) default here so the engine stays
model-agnostic. Constructing the scorer downloads the tokenizer, so this runs
only when NLI confidence is explicitly wired -- never as a side effect of
building the plain extractor.
"""

from __future__ import annotations

from config.settings import Settings
from infra.vllm.deployment import MEDGEMMA_4B, MEDGEMMA_4B_MODEL_ID
from nli import VllmNliScorer
from soap.nli_scorer import NliConfidenceScorer

# The Gemma chat-template yes/no prompt validated live in scripts/smoke_vllm_nli.py
# (10/10 on the golden set): MedGemma follows negations only inside its chat
# template. Kept here (not in VllmNliScorer) so the engine stays model-agnostic.
DEFAULT_NLI_PROMPT = (
    "<start_of_turn>user\n"
    "Read the Reference and the Claim. Does the Reference entail the Claim? "
    "If the Reference states or clearly implies the Claim, answer yes. "
    "If it contradicts or does not support the Claim, answer no. "
    "Answer with exactly one word, yes or no.\n\n"
)
DEFAULT_NLI_PROMPT_SUFFIX = "\n\n<end_of_turn>\n<start_of_turn>model\n"


def build_nli_confidence_scorer(settings: Settings) -> NliConfidenceScorer:
    """Construct the NLI confidence scorer from settings, with MedGemma defaults."""
    model_id = settings.model_id or MEDGEMMA_4B_MODEL_ID
    nli = VllmNliScorer(
        prompt=settings.nli_prompt or DEFAULT_NLI_PROMPT,
        prompt_suffix=settings.nli_prompt_suffix or DEFAULT_NLI_PROMPT_SUFFIX,
        model_id=model_id,
        tokenizer_id=settings.nli_tokenizer_id or model_id,
        vllm_base_url=settings.vllm_base_url or MEDGEMMA_4B.base_url(),
        api_key=settings.vllm_api_key or "",
    )
    return NliConfidenceScorer(nli)
