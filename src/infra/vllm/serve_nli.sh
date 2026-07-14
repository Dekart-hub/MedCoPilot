#!/usr/bin/env bash
# Конфиг развёртывания NLI-модели в vLLM (issue #13, FR-4).
#
# Поднимает отдельный сервис vLLM (OpenAI-совместимый HTTP API), к которому
# ходит src/soap/score/nli/VllmNliScorer. Это НЕ часть основного контейнера:
# приложению нужен только tokenizers, инференс целиком здесь.
#
# Модель и dtype НЕ захардкожены — задаются через env, потому что рабочий
# профиль зависит от железа (см. ниже). Два известных профиля:
#
#   [ПРОВЕРЕНО] движок валидирован на этом профиле (T4, 9/10 на golden):
#       NLI_MODEL_ID=Qwen/Qwen3-Reranker-0.6B VLLM_DTYPE=float16 \
#           bash serve_nli.sh
#
#   [ЦЕЛЕВОЙ, issue #13 — ЕЩЁ НЕ ЗАПУСКАЛСЯ] MedGemma; требует GPU
#   compute >= 8.0 (bf16): L4 / A100. На T4 (7.5) и P100 (6.0) MedGemma
#   (gemma3) не стартует вовсе — float16 ей запрещён (numerical instability),
#   bfloat16 не поддерживает железо, float32 переполняет shared memory. Так
#   что этот профиль надо ещё прогнать на Ampere+:
#       NLI_MODEL_ID=google/medgemma-4b-it VLLM_DTYPE=bfloat16 \
#           bash serve_nli.sh
#
# Env:
#   NLI_MODEL_ID        модель на HF (gated MedGemma -> нужен HF-логин + лицензия)
#   VLLM_DTYPE          float16 | bfloat16 | float32 (под железо и модель)
#   VLLM_HOST/PORT      адрес сервиса
#   VLLM_API_KEY        необязательный ключ для Authorization: Bearer <key>
#   VLLM_MAX_MODEL_LEN  длина контекста
set -euo pipefail

MODEL_ID="${NLI_MODEL_ID:?set NLI_MODEL_ID (e.g. Qwen/Qwen3-Reranker-0.6B)}"
DTYPE="${VLLM_DTYPE:?set VLLM_DTYPE (float16 / bfloat16 / float32)}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"

VLLM_ARGS=(
    serve "${MODEL_ID}"
    --host "${HOST}"
    --port "${PORT}"
    --dtype "${DTYPE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-logprobs 20
)
if [[ -n "${VLLM_API_KEY:-}" ]]; then
    VLLM_ARGS+=(--api-key "${VLLM_API_KEY}")
fi

exec vllm "${VLLM_ARGS[@]}"
