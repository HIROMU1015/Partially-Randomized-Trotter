# C2 LD2 fit/freeze preflight (2026-11-15 JST)

C2 branch `codex/pauli-ld2-price-freeze-20261115` starts at fixed C commit
`12372fa6b53c878dc5dd124e036572ba0cd82061`, which contains the saved exact fit
implementation. C and D2 worktrees/branches are not edited. The latest chat
explicitly authorizes this ten-record fit and four predictions, commit/push,
and handoff; it does not authorize holdout observation or execution.

## §10.5 duplicate check before scientific fit

Inspected related LD2 result/freeze history across local refs (last published
LD2 calibration handoff `5dac74959ed9ea49465d393fc792ef5b4727edba`), fixed
A `7a0c9468b2499fac9fa1cd279ae2980c9dab048e`, D2 plan
`043c7567451e8f502a1cf0f425d32efe0a50f5dc`, and C's price-route and PF
Stage1 model-freeze records. C's saved models are deterministic PF with
features one/q, not this LD2 dictionary with one/q/qR. D2 C_HANDOFF reports
fit_run=false and holdout_run=false. Related process-name inspection of this
Linux user's jobs found no running LD2 fit/freeze or B1 acquisition process.
This scope does not assert anything about inaccessible sessions or private
external runs. No equivalent frozen LD2 model was found in the inspected scope.

Remote read before C2 execution: D2 is the specified `5dac74959...`; C was
`04a68f5932f53377595b2d69bbf8f46e02e82a01` and is advancing separately.
The proposed C2 branch did not exist on the remote. Its new local worktree
contains no prior model output. No holdout observation file was read.

Reuse all 10 calibration records and D2's five saved C_CELL_CHECK results,
with 18 individual receipt locators. Match C_HANDOFF as authoritative to the
published per-cell records, sidecars, fixed physical ID, dictionary, seeds,
raw publication and execution SOURCE. Keep S66 path0→path1 as P2R66's ordered
parents. Eight fresh paths plus one derived paired path are not nine
independent paths; eighteen receipts are not eighteen fit observations.
B0/W are excluded.

Additional work needed: 2 exact OLS fits (5 cell means per axis, equal cell
weight), 4 exact predictions (m5/m6 × X/Y). No new compile, sampling, E[C], CI,
N or G. Metadata-only prepare reports ten records/eighteen receipts and rank 3.
The A admission module and C exact OLS run unchanged from their fixed Git blobs;
C's OLS/dependency bytes equal those at the C2 base.

D2's unchanged frozen_model_gate function body is checked with a saved fixed
plan seed-table lookup. This is a metadata consumer check, not execution of
D2's full request/compile pipeline. Pauli tolerance is unadopted; PF's 2%
rule is not applied. Independent holdout, actual LD2 order2 products and
nontrivial phase remain unevaluated/UNKNOWN. Stop after frozen-result handoff.
