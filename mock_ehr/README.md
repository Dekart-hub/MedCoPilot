# Mock FHIR R4 EHR

The publication contract uses the pinned HAPI FHIR JPA starter in
`compose.mock-ehr.yml`. It is a development-only service with an ephemeral H2
database and no authentication.

```bash
make mock-ehr-up
make mock-ehr-seed
make mock-ehr-live-test
```

The seed is an idempotent FHIR transaction containing the Patient, Encounter
and Practitioner referenced by the live publication test. Reset the service
with `make mock-ehr-down` followed by the three commands above.

The live test submits the generated document to HAPI's R4 validator, publishes
it twice, and verifies that conditional create plus identifier lookup resolve
both attempts to one stored Bundle.
