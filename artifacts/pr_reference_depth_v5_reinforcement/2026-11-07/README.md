# V5 depth guarantee reinforcement — 2026-11-07

[REPORT](../../../docs/research/pr_reference_depth_v5_reinforcement_20261107/REPORT.md) · [SCOPE](../../../docs/research/pr_reference_depth_v5_reinforcement_20261107/SCOPE.md)

**MANDATORY STOP.** Stage A PASS, Stage B PASS, Stage C MAINTAINED. Old total bounds and guaranteed depth reduction lower limit 40.995952643300015% remain valid under the reinforced fixed-model conditions. No new all18 total was computed, and inverse spans were not tightened to obtain a better result.

- [PREDECLARED.json](PREDECLARED.json): fixed before compilation: 5 inverse D structures × 4 reference contexts, 4 barrier-separated batch compiles; Bprime saved lower-chain inventory and conditional C.
- [STAGE_A.json.xz](STAGE_A.json.xz): full SOURCE forward/inverse order and actual binary angles, compiled native instructions, pre-final matrices/fences, Qiskit depth, transfer matrix and span comparisons for all 20 inverse records. Inverse and positive matrices differ; inverse spans never exceed the old upper bounds.
- [stage_A_raw.tar.xz](stage_A_raw.tar.xz): four original raw batch JSON, byte-preserving. STAGE_A also includes these records as structured rows.
- [STAGE_B.json.xz](STAGE_B.json.xz): all 144 saved D/frame record checks, 19 actual lower-bound names × 4 contexts = 76 dependencies; final local X-free chains, protected non-X delta/matrix, fence and CX-then-X reasoning. Pre-last internal X count is explicitly unrecorded, not filled with zero.
- [SOURCE_INVERSE_CHECKS.json](SOURCE_INVERSE_CHECKS.json): 13 basis SOURCE inverse-closure checks, installed Qiskit primary function hashes/locations, q1 and fused-lane mapping. The compiled native adjoint is not used as a SOURCE inverse.
- [SYNTHETIC_TESTS.json](SYNTHETIC_TESTS.json): 3 artificial tests, including CX deletions exposing X-X cancellation; two technical PassManager calls. Not scientific performance evidence.
- [STAGE_C.json](STAGE_C.json): old total bounds/comparison copied unchanged, applicability mapping for all18 q/r/bit/axis records. The conditional revised-bound code path was **not executed or qualified** by this run.
- [INPUTS.json](INPUTS.json), [ENVIRONMENT.json](ENVIRONMENT.json), [PRIMARY_SOURCES.json](PRIMARY_SOURCES.json), [SOURCE.json](SOURCE.json): scientific inputs, runtime and provenance.
- [STOP.json](STOP.json): final boundary. Old V5/pilot/Bprime/V3 and other Track STOPs remain untouched.

XZ files decompress to UTF-8 JSON and are byte-identical to their local uncompressed outputs. common.pkl, raw JSON working copies and uncompressed A/B remain ignored local files; the common input was decompressed from the fixed prior pilot, not regenerated.

Execution: initialize (including PREDECLARED), inverse_source_checks, stage_b, stage_a, synthetic_tests, stage_c, finish_stop. Stage B's inventory was made the union of all18/XY saved basis keys before final validation and compile; the final record set is 19 names. No SOURCE/compiler/gadget condition changed. Four independent A batches used four workers. No old runner, whole-native acquisition, sampling, quantum simulation, synthesis or all18 numerical recomputation occurred.

The user-provided task restates the two review issues. The separately named independent-review Markdown was not available in the base tree/known local path and is not claimed read. These are author checks, not a new independent review, external reproduction or formal verification.

After STOP only packaging, source/input hashes, documentation, static checks and publication were performed. Old SOURCE/raw/results/STOPs were preserved. Next scientific decision belongs to GPT/user; no automatic optimization or T synthesis follows.
