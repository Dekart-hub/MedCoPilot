# NLI-скорер через MedGemma в vLLM

Low-level движок оценки уверенности (FR-4): даёт `P(entailment)` в [0, 1] для
пары `(reference, inference)` по паттерну Qwen3-Reranker — читает логиты
токенов `yes`/`no` на одной позиции ответа, без генерации текста.

Модель обслуживается **отдельным сервисом vLLM** (OpenAI-совместимый HTTP API),
поэтому в наш контейнер torch не тянется. Для ручного клиента нужен только
опциональный extra `nli` с Rust-пакетом `tokenizers`:

```bash
uv sync --extra nli
```

Для gated-модели MedGemma сначала примите условия доступа на Hugging Face и
настройте `hf auth login` либо `HF_TOKEN`; токен не хранится в репозитории.

## Требования к железу (важно!)

MedGemma — это `gemma3`, а он на vLLM **не запускается на GPU с compute
capability < 8.0**. Цепочка ограничений замыкается намертво:

- `float16` — gemma3 запрещает сам (numerical instability, vLLM отказывает);
- `bfloat16` — требует compute ≥ 8.0 (Ampere+); Turing (T4, 7.5) и Pascal
  (P100, 6.0) его аппаратно не умеют;
- `float32` — в gemma3-ядрах запрашивает ~80KB shared memory на блок, а у
  T4/P100 лимит 64KB → `out of resource: shared memory`.

Итог: **бесплатные T4/P100 (Kaggle, Colab free) MedGemma не потянут.** Нужна
карта **compute ≥ 8.0 с bf16**: L4 (8.9), A100 (8.0) и т.п. (Colab Pro,
RunPod, vast). На такой карте поднимается простым конфигом ниже.

Это выяснено **эмпирически**, не в теории: попытка поднять
`google/medgemma-4b-it` на Kaggle (2×T4) реально упала — сперва
`ValueError: Bfloat16 is only supported on GPUs with compute capability of at
least 8.0. Your Tesla T4 GPU has compute capability 7.5`, а на fallback в
float32 — `out of resource: shared memory, Required: 81920, Hardware limit:
65536`. Поэтому проверку движка провели на Qwen (fp16), см. «Статус валидации».

## Запуск vLLM

Конфиг развёртывания — в пакете infra: [`src/infra/vllm/serve_nli.sh`](../../../infra/vllm/serve_nli.sh)
(модель и dtype задаются через env). Там же два профиля: проверенный
(Qwen/fp16) и целевой MedGemma/bf16 на Ampere+.

## Статус валидации

**MedGemma (целевая модель) — проверено на живом bf16-эндпоинте.** Движок
прогнан на `google/medgemma-4b-it` в vLLM (`--max-logprobs 20`) на golden-наборе
`golden_en.jsonl` — **10/10**, уверенное разделение (entailed → ~1.0,
contradicted → ~0.0). Весь путь `HTTP → allowed_token_ids → logprobs → softmax`
работает, форма ответа completions-API подтвердилась.

Ключевой момент — **промпт**: MedGemma это instruct-модель (gemma3), и на голом
completions-промпте (без чат-шаблона) она «залипает» на `yes` и не отрабатывает
отрицания — было **3/10**, отрицания шли с высоким score. Достаточно обернуть
инструкцию в чат-шаблон Gemma (`<start_of_turn>user … <start_of_turn>model`) —
и получаем 10/10. Сам движок (softmax по логитам yes/no) при этом не меняется;
калибруется только промпт (см. дефолт в `scripts/smoke_vllm_nli.py`). Live-пункт
Definition of Done issue #13 закрыт.

Для справки, ранее движок прогонялся и на `Qwen/Qwen3-Reranker-0.6B` (T4, fp16 —
Qwen не gemma3, потому заводится и на Turing): **9/10** с родным промптом Qwen,
единственный промах — пограничный score (~0.53) на явном отрицании. Это мотивирует
эскалацию на LLM при score в «серой зоне» ~0.3–0.7.

Проверка, что эндпоинт жив и умеет отдавать logprobs:

```bash
curl "$VLLM_BASE_URL/v1/completions" \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"google/medgemma-4b-it","prompt":"...\nAnswer:","max_tokens":1,"logprobs":20}'
```

## Переменные окружения (предложение, согласовать с DI)

| Переменная | Смысл |
|---|---|
| `VLLM_BASE_URL` | адрес vLLM, напр. `http://vllm:8000` |
| `VLLM_API_KEY` | необязательный ключ, переданный в `vllm serve --api-key` |
| `NLI_MODEL_ID` | id модели для запросов, напр. `google/medgemma-4b-it` |
| `NLI_TOKENIZER_ID` | что грузить локально для id токенов yes/no |

## Открытые вопросы (уточнить у автора issue #13, @12PAIN)

1. `calc_nli_score` сделан `async`, потому что текущий `ConfidenceScorer` и
   все сетевые адаптеры проекта асинхронны.
2. Промпт откалиброван на golden-наборе (10/10), но набор маленький (10 пар).
   Перед продакшеном стоит прогнать на большем/клиническом наборе и, при желании,
   вынести дефолтный промпт из smoke-скрипта в конфиг/DI.
