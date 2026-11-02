# H6 B0+I parallel compile — independent difference review

Date: 2026-10-15 JST. Reviewer: `/root/h6_pilot_independent_audit`.

**TECHNICAL PASS within the static/mock scope. No confirmed uncorrected defect was found in the reviewed difference. Execution remains unauthorized.** This continues the prior bounded-runner technical audit; it is not another scientific review or real parallel validation.

Baseline serial SOURCE: `b4c12d7a71dec57751ce21ebd434d797577ab450`. The final parallel SOURCE/contract commit was pending when this note was written; the exact reviewed working bytes are bound below. Final publication must bind these bytes to its actual SOURCE and source closure.

## Scope and corrected findings

Read the new `h6_b0i_parallel_compile_v1.py`, the runner hook difference, the new mock tests, harness difference, and complete 70-test result/log. No source or tests were edited by this reviewer in this continuation. No test, real process, fork, affinity operation, signal, scientific import, H6 input decoding, build, probe or compile was started by this reviewer.

Three concrete issues from the initial difference review are corrected:

1. **Fork ownership registration:** handle storage and registry insertion now occur before fork. The returned PID is stored before identity IO; failure to read identity retains the child in the cleanup registry. The cleanup path can reacquire and verify same-parent/UID identity before signaling.
2. **First STOP preservation:** an exceptional cleanup is recorded separately and cannot overwrite an earlier compile/IPC STOP. Real-backend emergency cleanup attempts owned kill/reap and FD closure. Cleanup wait exceptions continue to the KILL stage, and unconfirmed cleanup cannot produce COMPLETE.
3. **Artificial execution isolation:** the real `ForkBackend` rejects a synthetic grant, and the helper rejects its default real backend for synthetic execution. The runner also requires an explicit mock parallel hook for a synthetic run. The test harness independently disables real `os.fork`.

The pipeline still finishes all three Phase A/B cases before compile reservations. Fork workers inherit the six ready wrappers; they do not build or transfer new circuits. Exactly six unique axes and the existing three-build/six-compile budget remain. Parent and returned-result fingerprint/compiler/metric checks remain connected. Serial behavior remains available as the default path.

The child installs the parent-death guard with parent-PID checks before and after registration, then binds one approved CPU, private temporary/cache paths and a one-thread monitor. The parent drains bounded IPC, checks pipe EOF after wait4, compares native exit and PASS payload/owner identity, and keeps raw failure IPC in the cleanup evidence. Owned pidfd signaling rechecks identity around open/send; ESRCH alone is treated as already absent. Parent monitoring accounts for child RSS/AS/thread/affinity and aggregate parent-plus-child CPU.

## Evidence and limits

The parent-run bounded artificial result is **70/70 PASS**: new parallel 24, prior author 23, prior independent 23. The complete log and `artificial_result_attempt01.json` were inspected. Reported wall was 1.2548076249659061 seconds, peak RSS 32,505,856 bytes, scratch remaining zero, scientific imports/operations, real children, affinity changes and additional transpiles all zero. This review did not rerun the old bridge/scientific campaign.

These mocks test registration/failure paths, parent-death syscall/race handling, ownership/ESRCH, pipe overflow, native exit, aggregate CPU, cleanup continuation, all-A/B ordering, exact-six coverage and artificial serial/parallel output agreement. They do **not** establish real fork safety/performance, kernel parent-death behavior under actual driver SIGKILL, real compiled numeric equality, RSS sharing, actual speedup or native cleanup timing.

Resource observations and parent polling can lag; long C and blocking IO remain operational limits. Abnormal cleanup may explicitly remain UNKNOWN. Driver death can prevent terminal/reap receipts; missing evidence is not completion and exact exit time is not inferred. Real native exit/worker reap evidence, fresh source/input/profile identity, disjoint CPU/affinity, memory for every role, quota/output/temp/carry binding and explicit approval remain required. Quota conditions unsupported by the existing guard remain BLOCKED. No automatic retry or matched-B0 expansion is authorized.

The existing scientific provider, bridge, adapter and launcher have no working-tree difference in the inspected continuation. No new scientific claim or error policy is adopted. The preparation retains `approved=false`, `runtime_authorization=false`, `allowed_cpus=[]`, `execution_plan_sealed=false`, `absolute_launch_command=null`, `automatic_retry=false` and `next_stage_authorized=false` until separate explicit approval/fresh binding.

## Reviewed working bytes

| File | SHA-256 |
| --- | --- |
| `scripts/h6_b0i_parallel_compile_v1.py` | `4cf7b8b37737c1a6d1e49af6f049a3e7216bd2ebb88b55a0e5e46b5ee5896a96` |
| `scripts/h6_b0i_pilot_runner_v1.py` | `08b9f717fc5e6ecb3f917523eb9f150ff3b60fe525455ea6176055b46836c53f` |
| `tests/test_h6_b0i_parallel_compile_v1.py` | `02f18b141af6af7014d90b17cec46d4681177dbfbbfbbafc7e81a5b3b05c32f0` |
| `tests/run_h6_pilot_artificial_v1.py` | `bc274132e75a6349ba39840bf6bd0d49a49ab1f04f1eadedd0bbe4d5ac46de44` |

**Mandatory STOP. This review issues no execution approval or grant.**
