# H6 native B0+I pilot — independent technical audit v1

Date: 2026-10-15 JST. Reviewer: `/root/h6_pilot_independent_audit`.

Status: **TECHNICAL PASS within the static/artificial scope; real launch is conditional on explicit user execution approval and successful fresh binding.** This is a technical audit, not a new scientific review or execution authorization.

Reviewed actual runner SOURCE: **`b4c12d7a71dec57751ce21ebd434d797577ab450`**. Initial reviewed SOURCE was `25c45704d8e5dc1cca53b76faabbd53168b10699`; the final SOURCE adds only the private-temp fix and one related test described below.
The four new production files and the independent test file were compared with `git show SOURCE:path`; disk/Git bytes agree. Old scientific provider SOURCE remains `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7`; unchanged bridge/guards remain based on `20f89bc32b161ee22e4a1ed709f9e3414c19633a`.

## Scope and limits

Read the user attachment, common 2026-10-12 research instructions, AGENTS.md, PROJECT_MAP.md, the adopted 2026-10-15 scientific review, fixed bridge SOURCE `20f89bc32b161ee22e4a1ed709f9e3414c19633a`, fixed native SOURCE `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7`, and relevant supporting APIs with `git show` at that exact native SOURCE. No scientific library was imported. No real H6 input, array, scalar, signal, bias, allowance, headroom or compile result was computed or decoded. No native build/probe/compile was run. No old worktree, science source, grant, ledger or environment was modified. No subagent was spawned.

## Fixed API findings

- `build_deterministic_native` consumes unchanged prepared deterministic blocks and applies `p(-scalar*T, ancilla)` once after the complete PF schedule. The bridge supplies original constant plus discarded-tail identity once. Nuclear, one-body and retained-block scalar phases remain native behavior.
- `build_native_hadamard_wrapper` uses H / evolution / optional Sdg / H; cosine is Re(z), sine is Im(z). Ancilla index is the system-qubit count. It checks a fresh complete numeric evolution fingerprint before wrapping.
- The existing cost-only port and `compile_owned` consume/rebuild evolution for the second axis. Direct reuse would conflict with the new one-build-per-case requirement; the dedicated adapter must retain and compile its two measured wrappers instead.
- The existing saved-signal producer stores `independent_forward_signals`, `signal_discrepancies`, `state_discrepancies`, action counts and oracle counts in each signal-summary record. The fixed independent forward signal can be analytically rotated and compared without starting a new exponential-action path. This is a saved-state signal check, not full-vector independent replay or a full-operator certificate.
- Native control checks require four explicitly counted statevector probes per case: low branch, high branch and two unmeasured X/Y wrappers. Twelve probes across the three cases must precede any compile.
- Complete numeric fingerprint v2 includes global phase, numeric parameters, control state, recursive gate definitions, and qubit/classical-bit order. It is distinct from the bridge's prepared blueprint fingerprint. The adapter must persist and recheck actual wrapper fingerprints and returned compiler/fingerprint identity.
- Existing compile API: `transpile_and_measure_cost(wrapper, compiler, actual_circuit_fingerprint=fp)`; six metric fields are `rz_count`, `cx_count`, `rz_depth`, `cx_depth`, `total_depth`, `circuit_size`.

## Single implementation review and corrected findings

The reviewed runner completes Phase A for all three cases, then Phase B for all three, before reserving any compile. It rejects authorization/source/input/profile/registry changes before relevant scientific callbacks, keeps one-shot reservations consumed after failure, records partial evidence and first STOP, and keeps matched B0 deltas null. `NativeAdapter.validate` separately compares the corrected overlay and the saved independent forward signal with the native overlap, checks the low/high control branches, and checks cosine/sine probes. Wrapper fingerprints are rechecked before and after compilation, returned fingerprint/compiler identity and six integer metrics are checked, and the fixed scientific checkout is checked again at completion.

| Finding | Evidence and minimal correction | Final status |
| --- | --- | --- |
| CONFIRMED_DEFECT H6-P01: bridge/accounting schema mismatch | Fixed `saved_signal_overlay` has no `cell`, while `d0.accounting` unconditionally reads `record['cell']['id']`. The initial new pipeline passed that output unchanged and would stop at the first case. The runner now copies the signal and adds its own `B0I_p...` identity; the B0/request digest and saved record remain untouched. The independent fixture deliberately uses the real tiny bridge output without a fixture-only cell. | Corrected; independent success and separate-identity tests pass. |
| CONFIRMED_DEFECT H6-P02: missing final separate-provider identity gate | The first implementation checked the scientific checkout only at initialization, whereas final runner `runtime_gate` checks the separate runner root. `NativeAdapter.verify_bindings()` now checks provider disk bytes against source hashes and fixed Git bytes after all six compiles, without science imports/builds. | Corrected; direct production-method tiny-file mutation test and runner completion rejection pass. |
| CONFIRMED_DEFECT H6-P03: launcher exception/signal cleanup gaps | Author and independent review identified default TERM/INT behavior, identity acquisition outside cleanup protection, and a failing identity reacquisition skipping the bounded wait. The launcher now installs/restores handlers, protects acquisition/reacquisition, sends only to verified owned identities, performs bounded TERM/KILL waits, and preserves the original reason with `CLEANUP_UNCONFIRMED` when cleanup cannot be established. | Corrected in reviewed SOURCE; independent mock handler, ownership, ESRCH/non-ESRCH, TERM/KILL/wait/recorder ordering tests pass. |
| CONFIRMED_DEFECT H6-P04: temporary-directory fallback | Author identified that an externally specified future `output/temp` might not exist when Python first resolves a temporary directory, permitting fallback. The minimal followup now creates private `output/temp`, `temp/mpl` and `temp/cache` after authorization and exclusive output creation and before input/scientific callbacks. It sets only the current process's TMPDIR/MPLCONFIGDIR/XDG_CACHE_HOME and explicitly sets `tempfile.tempdir`. | Corrected in final SOURCE; added artificial test and the 23 affected author methods pass. No scientific/provider source changed. |

Guard completion also explicitly requires a sealed real execution plan, an absolute launch command, the adopted review commit/path, materialized provider/runtime hashes, and inherited AS/CPU hard-limit feasibility. A journal-byte cap is explicit. These changes do not change the scientific convention, numerical error policy or native lowering.

No confirmed uncorrected defect was found in this limited review. The review does not establish successful execution on real H6.

## Artificial evidence actually inspected

The parent executed the tests under `tests/run_h6_pilot_artificial_v1.py`; the independent reviewer did not start another test process. The complete harness and successful limited-run log were read as evidence.

- **23 independent methods PASS**, including real stdlib bridge output through the new pipeline, last-case Phase B failure with compile zero, source/input/profile/compiler mutation, one-shot/no-retry, wrapper mutation, missing/boolean metrics, saved independent-reference validation, direct production validation with two-amplitude artificial vectors, and launcher lifecycle mocks.
- **22 author integration methods PASS** in `artificial_result_attempt02.json`; the 45-method run has zero failures/errors, wall 0.6968632563948631 seconds, peak RSS 30,408,704 bytes, and zero remaining scratch bytes/entries.
- The final private-temp followup changes only `scripts/h6_b0i_pilot_runner_v1.py` and `tests/test_h6_b0i_pilot_runner_v1.py`. The independent reviewer statically read the exact two-file diff and added test; no further test process was started by this reviewer. **All 23 affected author methods PASS** in attempt03, zero failures/errors, wall 0.46755552664399147 seconds, peak RSS 29,884,416 bytes and zero remaining scratch bytes/entries. The already passing 23 independent and 17 bridge methods were retained without another run.
- The unchanged **17 bridge methods PASS** in attempt01. Therefore **63 distinct relevant methods passed**: final author 23, independent 23, bridge 17. The old 75-test campaign was preserved and not repeated wholesale.
- Attempt01 remains as failure evidence. Its three independent subtest failures were an incorrect fixture expectation: the top-level authorization SHA gate correctly rejected mutations before the expected field-digest gate. The fixture now updates only the top-level SHA while leaving the old per-field digest, so it challenges the intended gate. Author fixture errors involved serializing fabricated complex numbers to JSON. These were fixture defects, not evidence of real scientific execution or a broken authorization gate.
- The harness blocks NumPy/Qiskit/CuPy/PySCF/trotterlib/trottertracks imports, and the result reports no scientific imports, real scientific operations, real child processes, affinity changes or additional transpiles. Launcher process/pidfd/signal operations were mocks; no real process was spawned or signaled in these tests.

Evidence paths: `artificial_result_attempt01.json`, `artificial_attempt01.txt`, `artificial_result_attempt02.json`, `artificial_attempt02.txt`, `artificial_result_attempt03.json`, `artificial_attempt03.txt`, and `environment_profile_receipt_v1.json` in this private directory. The static environment receipt records 34,273 critical files / 1,503,620,794 bytes checked with zero mismatches, and 230 fixed-provider source files with zero mismatches. This is source/profile byte evidence, not old compiler output bit-equivalence.

## Final contract and readiness classification

The new draft was read as metadata only. It specifies exactly `(p5,q1)`, `(p10,q2)`, `(p15,q8)`, S2, T=0.8 and symmetric directional control; one snapshot load, three analytic overlays/scalar receipts, three evolution builds, six measured wrappers, six unmeasured probe copies, twelve statevector probes and at most six compiles. New SCF/DF/state/exponential action/trajectory/hardware shots and matched-B0 compile are zero. Five input-byte receipts, the separate 230-file provider closure and the 34,273-file runtime profile are materialized. Empirical allowance remains `certified=false`; full-H reference is not rotated.

The preparation has no identified issue requiring a new scientific decision. Remaining normal launch work is explicit execution approval plus fresh binding of exact SOURCE/contract/input/profile/carry, owned CPU/affinity, unused output/control/temp/one-shot namespaces, launch environment/command and current resource evidence. Fresh carry binding must preserve known/held/failed/unknown history, with reset/refund forbidden; unresolved historical counts must stay explicitly unknown.

The inherited metadata-reference paths are now explicitly bound to commit `20f89bc32b161ee22e4a1ed709f9e3414c19633a` and their full repository paths. Final publication binding must name the final reviewed SOURCE `b4c12d7a71dec57751ce21ebd434d797577ab450` and its updated runner SHA; this routine binding is the parent's final packaging step, not a new scientific calculation or review. The last inspected draft retained all unapproved/unsealed values.

Quota-enabled storage without a supported fresh effective quota receipt is **BLOCKED**, not made executable by approval alone. The default path intentionally stops on such a target; use of the present quota-disabled target still needs fresh effective evidence. If that condition cannot be satisfied, further technical connection is required rather than weakening the guard.

The numerical worker has one thread and no science pool. Persistent execution roles are launcher plus numerical worker; synchronous short-lived Git metadata processes may coexist, and the exit recorder runs after the worker. Python periodic monitoring cannot guarantee immediate interruption inside long C or blocking calls; returned staleness is rejected. The launcher supplies an independent wall/log poll, with one-second polling and bounded five-second TERM/KILL waits, so cap overshoot between polls is possible. Launcher SIGKILL or an unobservable native exit can leave terminal/native-wait evidence absent; absence is never completion, exact exit time stays null and cleanup may remain UNKNOWN. Artificial tests did not stress real long C, native RSS, hard-limit exits or actual SIGKILL.

Real H6 scalar, signal/bias/allowance/headroom, independent-reference presence/meaning on the bound input, physical native correctness, numeric fingerprints, compile feasibility and actual costs remain runtime-only and unmeasured. Such a runtime check can fail closed without changing the adopted science; artificial PASS is not its success proof. Full-vector independent equality and rigorous numerical certification are not claimed. PR winning is not a GO condition.

## Reviewed bytes

| Path | SHA-256 |
| --- | --- |
| `scripts/h6_b0i_pilot_runner_v1.py` | `960244168bdff8edeed352f591d0685a2d543ea56badfe37243dc0f23d77f5aa` |
| `scripts/h6_b0i_pilot_adapter_v1.py` | `a6f211390c783a3bdd901006f9ec9c41eef74a90896f9329293395ee2c98cfee` |
| `scripts/h6_b0i_pilot_launcher_v1.py` | `f93f679f00b4f3dc1b342b29ef7731a34d82ce5bfa9922856cb5722ec0035dc0` |
| `scripts/launch_h6_b0i_pilot_once_v1.sh` | `57d6a8ef40ecaaca1ad13f51bd0f0e5c6569fd72517110e610b8568eda794adb` |
| `tests/test_h6_pilot_independent_audit_v1.py` | `0d453ad1ed5c928cb1a840f28732ebf666d2c44675717fde47fc778c41abc364` |
| `tests/test_h6_b0i_pilot_runner_v1.py` | `ceadfb06032724cafbdd8732f0ac9c04d25ab54f94af1303279ea2489590ff79` |

Mandatory STOP remains required. Draft values stay `approved=false`, `runtime_authorization=false`, `allowed_cpus=[]`, `absolute_launch_command=null`, `execution_plan_sealed=false`, `automatic_retry=false`, `next_stage_authorized=false`.

**STOP. No execution authorization or grant was issued by this review.**
