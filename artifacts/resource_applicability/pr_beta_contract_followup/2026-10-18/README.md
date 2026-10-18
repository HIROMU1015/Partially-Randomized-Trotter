# Beta contract follow-up: B03 / B08 / B02-B06

Base audit: `3fd8fca966e6b5a6a972aa71de68b31a0129f5ab`.
Science snapshot: `f29eafbf4412193e8e8660d9f051889c2cba684b`.
B03 new SOURCE: `d87d217a2c4044fbe62e5aa4d3bc44bfb864ff00`.

[Package report](../../../../docs/research/pr_beta_contract_followup_20261018.md).

- `test_results.json` / `tests.log`: 9 artificial regression test methods, including 10 reused existing certification assertion cases. Actual accounting code; artificial preparation/cost types; not a full dependency-backed pytest run.
- `b08_investigation.json`: static generation recipes, metadata, 18 prior parameter matches, preparation mismatch, missing public worker payloads, and unapproved minimal follow-up proposals.
- `b02_b06_citation_metadata.json`: quoted prior 6/50 beta and 20/50 alpha counts. No budget-sum re-evaluation or retrospective guarantee assignment.
- `provenance.json`: 43 fixed input blobs and source identities. Prior 79-blob/38-check audit was reused, not rerun.
- `repository_checks.json`: prior/current full-manifest failures and new-entry-only validation, separately reported.
- `artifact_manifest.json`: output hashes and preservation checks; excludes its own hash.
- `STOP.json`: terminal technical scope. B08 remains scientifically unresolved. No follow-up grant.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tests/test_rpe_shot_underflow_20261018.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_pr_beta_followup_20261018.py
```

No molecular NPZ, private runtime or grant original is opened. No real
preparation/state/signal generation, saved-shot or cost reaccounting, scientific
reoptimization, circuit build/compile, or measurement simulation occurs.

The existing full-manifest validator has an inherited root-key schema mismatch;
its failure is preserved, not repaired here. Local commits only while external
publication authorization remains unconfirmed. No remote SHA is asserted.

MANDATORY_STOP. Technical tracing complete does not mean MODEL_MISMATCH resolved.
