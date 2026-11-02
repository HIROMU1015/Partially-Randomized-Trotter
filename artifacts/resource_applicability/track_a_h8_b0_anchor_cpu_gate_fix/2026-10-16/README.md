# H8 B0 CPU admission repair — no acquisition rerun

[Report](../../../../docs/research/track_a_h8_b0_cpu_gate_fix_20261016.md).

- Execution SOURCE: `db559a518214fd27fa96e2f1451911df05c4f4a8`.
- `execution_contract_v2.json`: same four cases/eight axes, call/resource caps. Raw 5% CPU threshold, one-second samples including SMT, no subtraction, no retry/reassignment.
- `current_CPU_diagnostic_v1.json`: current one-window diagnostic, NOT reconstruction of run01. CPU0 busy17/99 ticks; diagnostic itself on CPU43 with zero CPU-tick delta.
- `pre_grant_CPU_proposal_projection_v1.json`: one fixed three-window read-only proposal, projected to proposed/previous CPUs. No grant issued or CPU affinity changed.
- `source_profile_input_binding_v1.json`, `grant_INVALID_DRAFT_v2.json`: new sealed SOURCE, 15/265/34,273/8 source/provider/runtime/input byte checks. Invalid grant, no CPU allocation/launch command.
- `technical_validation_v1.json`: 11 new and 27 existing local unittest methods, ZERO scientific computation/compile.
- `preservation_receipt_v1.json`: original677 protected files/116 old worktrees unchanged.
- `publication_files_v1.json`, `verify_fix.py`: public byte verification only.

```sh
python3 artifacts/resource_applicability/track_a_h8_b0_anchor_cpu_gate_fix/2026-10-16/verify_fix.py
python3 -B -m unittest discover -s tests -p test_h8_b0_cpu_gate.py -v
```

The optional `scripts/probe_h8_b0_anchor_cpus.py --output NEW_FILE` reads counters for three seconds and proposes CPUs only. It cannot modify an existing grant, change affinity, launch science or replace the fresh gate. Fixed original busy/memory/physical-core conditions still apply. No automatic retry; new acquisition requires separate approval. Mandatory STOP after publication.
