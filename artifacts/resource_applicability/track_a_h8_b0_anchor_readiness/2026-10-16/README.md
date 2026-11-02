# H8 B0 new-anchor proposal — not executable or approved

[Plan and conditions](../../../../docs/research/track_a_h8_b0_anchor_acquisition_plan_20261016.md).

- `acquisition_plan_UNAPPROVED_v1.json`: S2 TRAIN p14q1/p21q1/p14q8 and HOLDOUT p21q8, both axes, 8 compile calls proposed. Resource caps and phase STOP boundaries fixed; real adapter/launcher not sealed.
- `saved_control_targets_v1.json`: exact copies of four saved B0 signals/independent-forward/full-H references, no recalculation.
- `input_source_receipt_v1.json`: 265 provider Git/disk checks and 7 existing input hashes; runtime inventory is historical, fresh full-byte gate still required.
- `grant_DRAFT_INVALID_v1.json`: invalid, approved=false, SOURCE/command null, CPU empty. Old grants cannot authorize this new scope.
- `technical_validation_v1.json`, `preservation_receipt_v1.json`: local artificial checks and old-worktree/protected-file preservation.
- `publication_files_v1.json`: public byte/SHA manifest, excluding itself.

```sh
python3 scripts/check_h8_b0_anchor_plan.py
PYTHONPATH=src/trotterlib python3 -m unittest discover -s tests -p test_h8_b0_anchor_plan.py -v
```

These commands only check the proposal. They cannot restore molecular objects, fit a model, create a valid grant or compile. Real execution requires a new sealed implementation, a separate explicit approval and a fresh resource/SOURCE/input/runtime gate. Acquisition, B1 fit, B2 S4 compile and C accounting each end in mandatory STOP.
