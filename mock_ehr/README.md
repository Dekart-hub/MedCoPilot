# External mock EHR

This directory provides a deterministic FHIR R4 boundary for local development.
The service is the official HAPI FHIR JPA starter image, launched separately from
MedCoPilot by `compose.mock-ehr.yml`. It has no authentication and must not be
treated as a production deployment.

## Start and seed

```bash
make mock-ehr-up
make mock-ehr-seed

# In the MedCoPilot .env:
EHR__ENABLED=true
EHR__BASE_URL=http://localhost:8080/fhir

make dev
```

The HAPI UI is available at <http://localhost:8080/> and its FHIR endpoint at
<http://localhost:8080/fhir/>. The image is pinned to
`hapiproject/hapi:v8.10.0-2` so fixture behavior is reproducible.

HAPI uses an ephemeral H2 database in this configuration. Reapplying either
bundle is safe (the transaction uses deterministic ids and `PUT`), but removing
the container resets all resources.

## Fixture boundary and label leakage

`pre-visit-bundle.json` is the only bundle loaded by the default seed command.
It contains demographics and resources already known before the linked
encounter. Every readable clinical resource has this FHIR tag:

```text
system = urn:medcopilot:fixture-phase
code   = pre-visit
```

The adapter returns only resources with that tag. It also rejects any Condition
linked to the current Encounter, even if it was incorrectly tagged. The
`post-visit-bundle.json` contains the completed encounter and gold diagnosis; it
exists to test the boundary and is loaded only with:

```bash
make mock-ehr-post-visit
```

Even after that bundle is loaded, the app's context endpoint must not return the
post-visit diagnosis.

Run the live contract after pre-visit seeding, and again after loading the
post-visit bundle:

```bash
make mock-ehr-live-test
make mock-ehr-post-visit
make mock-ehr-live-test
```

The live contract covers the direct FHIR adapter and the FastAPI endpoint. It
expects both raw Conditions to coexist after the second seed, while app context
continues to expose only the historical pre-visit Condition.

## Stable linkage and ICD-10

`fixtures/manifest.json` is the single explicit mapping:

```text
case -> Dialogue/{id} -> Encounter/{id} -> Patient/{id} -> Condition/{id}
```

`condition_ref` is always a FHIR reference such as
`Condition/mock-condition-tension-headache-001`. ICD-10 belongs inside the
Condition's `code.coding`:

```json
{
  "system": "http://hl7.org/fhir/sid/icd-10",
  "code": "G44.2",
  "display": "Tension-type headache"
}
```

The initial fixture is intentionally a small, project-owned synthetic smoke
case. Dataset importers for curated PriMock57 cases and Synthea-generated
patients can extend the same manifest without changing the application API.

## Workflow

1. `GET /api/v1/ehr/dialogues/{dialogue_id}/context` reads bounded context.
2. `POST /api/v1/reports` generates and stores a draft report.
3. `POST /api/v1/reports/{report_id}/approve` records clinician approval.
4. `POST /api/v1/reports/{report_id}/ehr-sync` conditionally creates one FHIR
   `DocumentReference`.

Sync uses the report ID as a stable FHIR identifier and sends
`If-None-Exist`, while the local workflow also returns an already-synced result
without issuing another request. Both layers make retries idempotent.

## Reset and troubleshooting

Reset to a pristine pre-visit state:

```bash
make mock-ehr-down
make mock-ehr-up
make mock-ehr-seed
```

Stop and remove the service:

```bash
make mock-ehr-down
```

If Docker Desktop returns an HTTP 500 before the image pull begins, verify that
both client and server are available with `docker version`, restart Docker
Desktop, and retry `make mock-ehr-up`. Successfully downloaded layers are cached.
