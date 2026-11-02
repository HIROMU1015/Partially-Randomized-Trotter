# H8 B0 run01: STOP before science at fresh CPU gate

[Report](../../../../docs/research/track_a_h8_b0_anchor_run01_stop_20261016.md).

- `attempt_STOP_v1.json`: scope, approval basis, pinned SOURCE, CPU assignment, observed STOP and zero science calls. Failing CPU samples and launcher native exit code remain unknown.
- `CPU_selection_before_launch_v1.json`: earlier two-second idle selection, distinct from the failing one-second fresh check.
- `launcher_error_projection_v1.json`: original stderr hash/bytes and relative stack projection. Raw private paths excluded; original stderr unchanged.
- `source_profile_input_integrity_v1.json`: post-STOP read-only source14/provider265/runtime34,273/input8 check. No fresh-gate retry.
- `preservation_receipt_v1.json`: original 676 protected files and 115 old worktrees unchanged. One new acquisition grant issued; its original bytes preserved separately by grant SHA.
- `publication_files_v1.json`, `verify_saved_stop.py`: public byte manifest and saved-state consistency check.

```sh
python3 artifacts/resource_applicability/track_a_h8_b0_anchor_run01_stop/2026-10-16/verify_saved_stop.py
```

No driver or worker started, no output/control/science lock/fresh receipt was created, no scientific reservation or compile occurred. No automatic retry or CPU reassignment. Old B1 STOP and V2 NO_GO remain; B1/B2/C are not authorized. Mandatory STOP after publication.
