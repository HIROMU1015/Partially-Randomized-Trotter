# Fixed comparison contract: one saved DF H4 case

This independent validation is authorized by the current user instruction, not a restart of any historical grant/one-shot/STOP. Fixed before numeric execution.

Case: Linear H4 / 0.90 Angstrom / STO-3G / DF rank12 / B2 / S2 / L_D=3 / q=1 / r=4 / K=4 / T=delta=0.8. Exact candidate `B2-rank3-q1-r4-K4`. Stored-order deterministic positions 0,1,2; randomized 3..11. No reranking or candidate selection.

P1: `2c865c66171fc9d1972b67c1257c57e5684fe26c`. Historical reference producer: `049e69919af16ad29a67a217dc7a407d6b1754a6` (original array freeze); dataset identity SOURCE `5aa6c3685afb9c3afcfbdc36f80244da0a7fe525`; later candidate execution `d0726e946f89f8c30a08d383fb8ff48b363629ec`; donor `821cc0f6ced57168293ea166644024867b064ffb`. run05 `6d365257770e99022b91d6a38dbee49ee0077503` is a separate stopped lineage stage. These are distinct roles, not interchangeable snapshots.

## Exact data and comparisons

The same read-only, hash-qualified NPZ supplies the P1 adapter and historical stored reference arrays. Public file SHA is `06672860ed8f9298053b4d84693fbab6d13e38c3166c27dbabcc161f33673c3d`. Full32-array byte-hash audit is reused; only whole-file NPZ/freeze SHA is refreshed. Runtime loads use allow_pickle=False and immutable arrays.

- Actual H4GeometryAdapter.bind_core/input invokes the fixed same-checkout DFHamiltonian constructor once. Comparing forwarded arrays alone is WIRING/METADATA only. The input StageResult remains UNQUALIFIED as in P1; no composer/CLI scientific mode is enabled.
- Existing df_linear_operator(backend=python) reconstructs the represented fermionic Hamiltonian on the full 256-state Fock basis. One matmat call on identity256 invokes at most256 existing matvecs. Compare its matrix against independently stored original H; no new scientific kernel is implemented.
- Compare corrected one with stored hspin+correction, nuclear passed once, all lambdas/G positions unchanged. Explicit partition record is supplied without any ranking. Its abs_lambda weight is transport metadata only, never the finite-RTE one-norm.
- Existing uncontrolled/controlled identity evolution operators, once each, compare the nuclear scalar phase with analytic cos/sin and expected diag(I,U) form. This checks identity-only phase/control, not a full randomized controlled step.
- One batch checks original state norm, sector embedding, bit reversal, phase convention, stored energy and H-state residual. No eigensolver, new ground state, evolution simulation or sampling. Reuse the reconstructed matrix; do not re-call the operator.

## Fixed tolerances

| Quantity | Absolute | Relative | Basis |
| --- | ---: | ---: | --- |
| Full H and corrected one-body | 1e-12 | 1e-12 | Same binary64 input, thirteen represented terms, no scientific approximation change; consistent with old1e-12 algebra/state gate |
| State norm, phase, bit/sector mapping | 1e-12 | 0 | Original inputs.py gates |
| H-state residual and stored energy | 1e-9 | 0 | Original frozen-state residual/energy gates |
| Nuclear identity phase/control | 1e-14 | 1e-14 | Core relative parameter tolerance and binary64 identity formula |
| Hash/identity/order/partition | exact | exact | Provenance comparison |

No threshold is widened after observing outputs. Elementwise matrix acceptance is abs(a-b)<=atol+rtol*abs(reference). State residual uses Euclidean norm, not an eigensolve.

## Individual inconclusive items

Full P1 preparation calls build_one_body_gaussian_block/build_df_blocks -> U_to_qiskit_ops_jw -> BogoliubovTransform, which constructs a real circuit. The current instruction prohibits that build, so neither full P1 preparation nor old prepare is called. S2 half/reverse order can be inspected statically but numerical S2 equivalence is INCONCLUSIVE.

K=4 paired finite-law/r=4 short-step meanings are identified statically. The candidate's exact tail components/lambda are not available without the unpermitted preparation or a qualified saved config. Do not substitute the partition proxy, infer lambda from saved normalization, invent a tail/config or evaluate an unrelated tau. Finite-law numerical equivalence is INCONCLUSIVE.

No qualified explicit stored event sequence exists in verified SERVER_A paths/public projection. No new random trajectory/seed is generated; event request remains INCONCLUSIVE. The original finite-RTE complex signal and its normalization are deterministic saved quantities;32 trajectory cost/SE samples are a separate reference class. The public projection gives deterministic bias/normalization but not the full complex signal/events. No candidate-signal comparison is performed without its qualified tail/config.

## Call/resource ceiling and stop

One source/one input/one candidate, one process. Constructor/input/binding/operator and each identity-phase API once; one identity256 matmat, maximum256 internal Python matvecs. State/scalar comparisons one batch. Full preparation, finite-RTE numeric configuration without verified tail, event execution, circuits, compile, sampling, eigensolver/SCF, N/G/SE/shot calls: zero.

Task-only limits: 16GiB address space,300 CPU seconds,600 wall seconds, largest matrix512, private numeric output64MiB. Existing BLAS defaults; no global scheduler/cache/venv/resource setting changes. Limits bound this finite one-case audit, not shared research.

Input/SOURCE/identity failure or forbidden-call attempt stops the numeric process. Numerical mismatch stops dependent evaluations, records MISMATCH and leaves data unchanged. Independent qualified checks may finish. No real-API retry or kernel edit to force MATCH. Partial results are preserved and published if safe.

MATCH_INDEPENDENT requires original saved numerical data or a separate mathematical reference. Forwarding to the same API is MATCH_WIRING_ONLY. Metadata equality is MATCH_METADATA_ONLY. This contract is a validation plan, not an approval certificate or a new grant.

Reference-class clarification after execution: the original private pre-execution contract and its digest remain unchanged. This public wording distinguishes deterministic finite-RTE signal from sampled compile costs. Numerical comparisons, tolerances, call caps and the three unperformed items were not changed or rerun.
