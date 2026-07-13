# NLI-скорер через MedGemma в vLLM

Low-level движок оценки уверенности (FR-4): даёт `P(entailment)` в [0, 1] для
пары `(reference, inference)` по паттерну Qwen3-Reranker — читает логиты
токенов `yes`/`no` на одной позиции ответа, без генерации текста.

Модель обслуживается **отдельным сервисом vLLM** (OpenAI-совместимый HTTP API),
поэтому в наш контейнер torch не тянется — только `tokenizers` локально.

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

Движок (`calc_nli_score`) проверен на живом vLLM на **`Qwen/Qwen3-Reranker-0.6B`**
(T4, fp16 — Qwen не gemma3, потому и заводится): весь путь
`HTTP → allowed_token_ids → logprobs → softmax` работает, форма ответа
подтвердилась. На golden-наборе `golden_en.jsonl` с родным промптом Qwen —
**9/10**; единственный промах — кейс с явным отрицанием («no history of
diabetes» vs «has diabetes»), где score вышел пограничным (~0.53), а не
уверенно неверным. Это мотивирует эскалацию на LLM при score в зоне ~0.3–0.7.
Качество именно на MedGemma замеряется отдельно, когда будет bf16-эндпоинт.

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
| `VLLM_API_KEY` | ключ, переданный в `vllm serve --api-key` |
| `NLI_MODEL_ID` | id модели для запросов, напр. `google/medgemma-4b-it` |
| `NLI_TOKENIZER_ID` | что грузить локально для id токенов yes/no |

## Открытые вопросы (уточнить у автора issue #13, @12PAIN)

1. Готового vLLM-эндпоинта нет — конфиг выше **предложение**; финальный образ
   MedGemma и хост согласовать.
2. `calc_nli_score` сделан `async` (сетевой вызов + параллельность claim'ов
   через `asyncio.gather`), хотя в контракте сигнатура синхронная — подтвердить.
3. Разметка пары (`<Reference>/<Claim>`) и текст `prompt`/`prompt_suffix` —
   черновые, подбираются под MedGemma по golden-набору `golden_en.jsonl` (рядом).
