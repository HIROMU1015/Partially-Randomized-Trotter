# H6 matched accuracy/resource v2 execution evidence

This directory binds the separately authorized single execution to frozen science source `54986a9f54515c8c2289e38fa16fd81e2b6008c2`. Grant-only commit: `2157da1d208a04102166fa1ae3c154490546c90f`. Preparation and manifest remain immutable historical records.

- [Execution seal and scope](../../../../docs/research/track_a_h6_matched_accuracy_execution_seal_v2.md)
- [Authorization](authorization_v2.json): approved once; no retry/resume, old v1 rerun, H8, or additional science.
- [Local preflight](local_preflight_v2.json) and [remote preflight](remote_prelaunch_verification_v2.json): source216/input33/environment/resources; remote253 path bytes.
- [Initial host process observation](process_start_observation_v2.json): CPU/time/AS receipts, outside the scientific precision evidence.
- [Partial checkpoint](running_checkpoint_v2.json): immutable files from completed workers, while the overall run is still in progress. This is not the final result.
- [Preservation baseline](baseline_before_execution_v2.json): existing tracked and nonignored untracked bytes, including dirty documents. Nothing from Track B is staged.
- [Publication-only helper](finalize_matched_run_v2.py): waits for the launched coordinator's PID/start identity to end, runs only the existing static audit, preserves raw/missing evidence, publishes exact allowlisted files and document additions, and verifies independent GitHub refetch. It never starts scientific work or changes the frozen comparison. If the expected HEAD/index/old bytes change, it refuses to overwrite them.

After the execution ends, final evidence will be recorded in this directory and a separate result document. Those files do not yet exist in this checkpoint. Publication progress/failure receipts are local orchestration records; completion must be established by the final GitHub result and remote verification, not by this README.

Mandatory STOP after publication; `H6_NOT_AUTHORIZED`, `DRAFT_NOT_AUTHORIZATION`, `next_stage_authorized=false`. No subsequent calculation is authorized by the helper or this evidence directory.
