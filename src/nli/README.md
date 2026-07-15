# NLI scorer via MedGemma in vLLM

Low-level confidence engine (FR-4): returns `P(entailment)` in [0, 1] for a
`(reference, inference)` pair, following the Qwen3-Reranker pattern -- it reads
the logits of the `yes`/`no` tokens at a single answer position, without
generating any text.

The model is served by a **separate vLLM service** (OpenAI-compatible HTTP API),
so no torch is pulled into this package. Locally only `tokenizers` is needed, to
resolve the `yes`/`no` token ids. For the gated MedGemma checkpoint, accept the
license on Hugging Face and provide an `HF_TOKEN` (or `hf auth login`); the token
is never stored in the repo.

## Hardware requirement

MedGemma is a `gemma3` model, which will not start on vLLM on a GPU with compute
capability < 8.0: `float16` is numerically unstable (vLLM refuses it), `bfloat16`
needs compute >= 8.0 (Ampere+), and `float32` overflows gemma3's shared-memory
budget on older cards. So free T4/P100 (Kaggle, Colab free) cannot run it -- a
compute >= 8.0 card with bf16 (L4, A100, ...) is required. This is empirical, not
theoretical.

## Launching vLLM

The deployment config lives in the infra package:
[`src/infra/vllm/deployment.py`](../infra/vllm/deployment.py). `MEDGEMMA_4B`
serves the model in `bfloat16` and raises `--max-logprobs` to 20 so this scorer
can read the `yes`/`no` token logits.

## Validation status

**MedGemma (target model) -- verified live on a bf16 endpoint.** The engine was
run against `google/medgemma-4b-it` in vLLM (`--max-logprobs 20`) over
`golden_en.jsonl`: **10/10**, clean separation (entailed -> ~1.0, contradicted ->
~0.0). The whole path `HTTP -> allowed_token_ids -> logprobs -> softmax` works.

The key factor is the **prompt**: MedGemma is an instruct model (gemma3), and on
a bare completions prompt (no chat template) it sticks to `yes` and ignores
negations -- **3/10**. Wrapping the instruction in the Gemma chat template
(`<start_of_turn>user ... <start_of_turn>model`) yields 10/10. The engine
(softmax over the `yes`/`no` logits) is unchanged; only the prompt is calibrated
(see the default in `scripts/smoke_vllm_nli.py`).

## Environment variables (sanity script)

| Variable | Meaning |
|---|---|
| `VLLM_BASE_URL` | vLLM address, e.g. `http://localhost:8001/v1` |
| `VLLM_API_KEY` | optional key passed to `vllm serve --api-key` |
| `MODEL_ID` | model id for requests, e.g. `google/medgemma-4b-it` |
| `NLI_TOKENIZER_ID` | what to load locally for the `yes`/`no` token ids |

## Open questions (issue #13, @12PAIN)

1. `calc_nli_score` is `async` because the surrounding scorer and all networked
   adapters in the project are async.
2. The prompt is calibrated on a small golden set (10 pairs). Before production,
   validate on a larger clinical set and consider moving the default prompt out
   of the sanity script into config/DI.
