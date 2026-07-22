# T29 phase 1 — implementation notes

Branch `feature/29-icd-resolution` (from `main`), issue #77. Scope agreed for
phase 1: **always select the top candidate** — no score/margin thresholds, no
`AMBIGUOUS` behaviour — but the full contract shapes (status, candidates,
classifier version) ship now so phase 2 is logic-only. English catalog only;
Russian is future work.

## What was built

- Domain (`soap/soap.py`): `IcdResolutionStatus` (`resolved` / `ambiguous` /
  `not_found` — `ambiguous` reserved), `IcdCandidate` (code, name, rank, raw
  `bm25_score`), `IcdResolution`; `AssessmentClaim.icd_resolution`.
- `icd/resolver.py`: `IcdResolver` port (resolve + catalog `entry()` lookup),
  `NullIcdResolver`, manual-entry validation (`validate_manual_icd`) with
  `UnknownIcdCode` / `InactiveIcdCode` / `IcdTitleMismatch`.
- `icd/bm25_resolver.py`: `Bm25IcdResolver` — Okapi BM25, code-level dedup
  (synonym rows), deterministic `(score desc, code asc)` ordering, inactive
  entries excluded from the index, dictionary version stamped on every result.
- `icd/tokenizer.py`: Snowball stemming + clinical abbreviations + stopwords
  that keep `with`/`without`/`no`/`not` (ported from the classifier-branch
  retrieval work).
- `icd/bm25_coder.py` (T10) is now a thin top-1 wrapper over the resolver —
  one index, one tokenizer for both ports.
- Extractor: `resolver` parameter; fills `icd_resolution` and mirrors
  `selected` into `icd`. `coder` still works (resolver wins if both given).
- Persistence: 3 nullable columns on `soap_claim` and `soap_corrected_claim`
  (`icd_status`, `icd_classifier_version`, `icd_candidates` JSON), migration
  `0008_icd_resolution`, both repositories round-trip the resolution.
- API: `icd_resolution` additively in claim payloads (rank only, no raw
  score); manual ICD validated in the correction use cases against the same
  catalog → 422; `classifier_url` in input payloads ignored, always
  server-derived.
- Settings: `ICD_DICTIONARY_PATH` (unset ⇒ bundled sample), `ICD_TOP_K` (10).
  `scripts/fetch_icd.py` now writes a `.meta.json` sidecar (`icd10cm-2024-cms`).
- Tests: `tests/test_icd_resolver.py` on a hand-written fixture catalog
  (`tests/fixtures/icd_catalog.json`) — synonyms, ties, inactive, dedup,
  abbreviations, validation; extractor resolution tests; API 422 +
  canonicalisation tests; DB round-trip extended. 269 passed, 19 DB-gated
  skips locally.

## Deviations from the reviewed plan

1. **Candidates are a JSON column, not child tables.** The plan proposed
   `soap_claim_icd_candidate` (+ corrected twin). On reading the ORM,
   citations are already stored as a JSON column on the claim row — the JSON
   column mirrors the codebase's own pattern, needs no joins, and T30/T37 read
   candidates per-claim anyway. One migration, two columns fewer moving parts.
2. **Existing test payloads had to change.** `test_correction_api` /
   `test_quality_api` posted `"Migraine"` for G43.9 with fake URLs; catalog
   validation now (correctly) rejects that, so the tests use the canonical
   `"Migraine, unspecified"`. This is the intended contract change, not
   collateral.
3. **Strict title matching (per ticket: invalid code/title → 422).** The UI
   lets a doctor free-type the name; with validation on, anything non-canonical
   is a 422. If that proves harsh in practice, the fallback discussed in the
   plan (ignore the client title, store canonical) is a one-line change in
   `validate_manual_icd`.

## Known consequences / edge decisions

- **A doctor's manual edit drops the stored resolution for that corrected
  claim** (the payload can't carry one — client-supplied candidates would be a
  tamper vector). The immutable source report keeps the full audit trail;
  `start correction` copies resolutions via `deepcopy`.
- "Always return a code" is not literal: empty/out-of-vocabulary text stays
  `not_found` with `selected = null` — we do not invent codes for garbage
  (this is the ticket's NOT_FOUND row, minus the score threshold).
- `IcdCoder`-only setups (no catalog wired) skip manual-ICD validation; the DI
  wiring always provides the resolver, so every API path validates.
- The bundled sample's WHO-style names differ from CMS ICD-10-CM wording for
  some codes; when `ICD_DICTIONARY_PATH` switches to the CMS file, manual
  entries must match CMS titles (the golden rule: the catalog you resolve with
  is the catalog you validate with).

## Before merge

- [ ] Run the 19 DB-gated tests against Postgres (docker was down locally):
      `docker compose up -d postgres && DATABASE_URL=postgresql+asyncpg://medcopilot:medcopilot@localhost:5432/medcopilot uv run pytest tests/ -q`
      — this exercises migration `0008` for real.
- [ ] Decide DoD checkboxes on issue #77: phase 1 closes everything except
      «близкие top candidates → AMBIGUOUS» (phase 2) and «тесты на русском
      тексте» (deferred by decision d. 2026-07-22).
