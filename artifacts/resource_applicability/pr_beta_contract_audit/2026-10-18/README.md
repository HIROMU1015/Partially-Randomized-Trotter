# PR beta contract audit — A-1 through A-5

Fixed science snapshot: `f29eafbf4412193e8e8660d9f051889c2cba684b`.
Audit base: `e62ddb59173d100c113be7a33581515d78f152ee`.

[Japanese report](../../../../docs/research/pr_beta_contract_audit_20261018.md).

- `technical_validation.json`: 79 fixed blob identities, saved metadata checks, 38 artificial checks, findings, parameter-ID physical-grid joins with a preparation mismatch.
- `beta_contracts.json`: 50 saved WP04/WP01-v2/WP11 contracts / 904 round rows; separate original 10 allocation scenarios. Saved values only, no shot or cost rederivation.
- `artifact_manifest.json`: hashes of this audit's outputs and auditor. Does not identify historical scientific inputs as regenerated.
- `STOP.json`: terminal audit scope; numerical follow-ups remain unauthorized.
- `repository_checks.json`: inherited full-manifest schema failure, identical at base/current; passing new-entry-only check. This is not full-manifest PASS.

Reproduce with Python standard library and this repository's Git object database:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 scripts/audit_pr_beta_contracts_20261018.py
```

`python3 scripts/check_validation_manifest.py` currently fails on preexisting
extension root keys, also present at the audit base. The core-schema projection
of the new entry passes; see `repository_checks.json`. Existing entries and the
validator were preserved.

No scientific module is imported. Molecular NPZ, private runtime, live grants,
sampling, signal generation, real-candidate optimization, saved-shot evaluation,
cost reaccounting, Monte Carlo, circuit build, and compilation are outside this
auditor. Budget sums and comparisons of stored values are technical checks.
The artificial checks intentionally reproduce an underflow exception; their
passing status does not assert that the underlying implementation defect is fixed.

MANDATORY_STOP. No automatic next phase, retry of a scientific call, or scientific
claim adoption. Source/result commit and remote publication are reported separately.
