# H8 B0 anchor execution readiness — NOT AUTHORIZED

[Readiness and approval scope](../../../../docs/research/track_a_h8_b0_anchor_execution_readiness_20261016.md).

- Execution SOURCE: `572a337d0dc3fa78b2fd6b25ace6a37845db384e`; four fixed cases, eight S2 axes, max four workers/two waves.
- `execution_contract_v1.json`: exact contract derived from the unchanged acquisition proposal.
- `source_input_profile_binding_v1.json`: source14/provider265/runtime34,273 byte checks, eight inputs (seven saved scientific/metadata inputs plus font seed), invalid private config digest. No scientific restoration/execution or fresh resource gate.
- `grant_INVALID_DRAFT_v1.json`: source-bound but unapproved/invalid; CPU empty, command null. Cannot launch.
- `technical_validation_v1.json`: 17 new unittest methods, 10 unchanged plan methods, eight two-qubit provider checks. Four artificial compile axes; ZERO real H8 compile axes.
- `preservation_receipt_v1.json`: original 673 protected files and 114 worktrees unchanged.
- `publication_files_v1.json`: public byte manifest; `verify_readiness.py` checks this public bundle only.

```sh
python3 artifacts/resource_applicability/track_a_h8_b0_anchor_execution_readiness/2026-10-16/verify_readiness.py
python3 -B -m unittest discover -s tests -p test_h8_b0_anchor_execution.py -v
```

No approval is implied by checker PASS. After separate explicit approval, a fresh six-physical-core/176.25-GiB/quota/PSI/OOM/SOURCE/input/runtime gate is mandatory. Acquisition ends in STOP even on success; B1 fit/holdout, B2 S4 compile and C accounting need independent approvals. No old B0 costs or grants are promoted.
