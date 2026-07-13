# NLI-скорер через MedGemma в vLLM

Low-level движок оценки уверенности (FR-4): даёт `P(entailment)` в [0, 1] для
пары `(reference, inference)` по паттерну Qwen3-Reranker — читает логиты
токенов `yes`/`no` на одной позиции ответа, без генерации текста.

Модель обслуживается **отдельным сервисом vLLM** (OpenAI-совместимый HTTP API),
поэтому в наш контейнер torch не тянется — только `tokenizers` локально.

## Запуск vLLM с MedGemma

```bash
# отдельный сервис/контейнер с GPU; НЕ часть основного образа
vllm serve google/medgemma-4b-it \
    --host 0.0.0.0 --port 8000 \
    --api-key "$VLLM_API_KEY" \
    --max-logprobs 20
```

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
