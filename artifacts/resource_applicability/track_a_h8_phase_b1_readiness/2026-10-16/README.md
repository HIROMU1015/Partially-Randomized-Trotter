# H8 Phase B1 preparation — STOP, no numerical fit

[Report](../../../../docs/research/track_a_h8_phase_b1_anchor_readiness_20261016.md).

- `anchor_registry_v1.json`: 32 copied saved measured axes, provenance flags and blocking reasons. B0 0/16 eligible; B0I 16/16 eligible. No predicted costs.
- `anchor_gate_v1.json`: full B1 STOP; no actual design-rank, fit or residual calculation.
- `source_evidence_v1.json`: 82 existing fixed-source/public/private-receipt hashes; private paths/payloads excluded. Historical runtime binding only.
- `S4_eligibility_metadata_v1.json`, `B2_selection_v1.json`: saved eligibility bools and previously registered four-axis selection. Predictions/model SHA null; compile not authorized.
- `execution_contract_DRAFT_v1.json`, `grant_DRAFT_INVALID_v1.json`: invalid proposals, approved=false, SOURCE/command null, CPU empty. No fit runner or grant issuer.
- `technical_validation_v1.json`, `preservation_receipt_v1.json`: primary-agent local artificial tests and preservation audit.
- `publication_files_v1.json`: SHA/byte manifest for public evidence and guard code.

Read-only verification from repository root:

```sh
python3 scripts/check_h8_phase_b1_readiness.py
PYTHONPATH=src/trotterlib python3 -m unittest discover -s tests -p test_h8_phase_b1_readiness.py -v
```

Checker PASS means the STOP bundle is consistent. It does not mean scientific execution is ready or authorized. Separate approval and resolved provenance are required; mandatory STOP after publication.
