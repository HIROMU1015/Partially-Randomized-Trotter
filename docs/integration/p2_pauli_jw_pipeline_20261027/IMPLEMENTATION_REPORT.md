# P2 Pauli-JW pipeline implementation

LOCAL technical implementation only. Started 2026-10-27; completed 2026-10-28 JST. New branch:
`codex/p2-pauli-jw-pipeline-integration-20261027`. Base **`00958ae1945c780718b2deef07d5dab1096a5cd8`**, the latest referenced H6 formal-handoff branch tip observed on entry.
Summary `b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b` remains the unchanged representative saved-result entry.
Choosing the later base retains the already completed H6 adapter and existing Pauli dependencies in this checkout;
no whole-branch merge, kernel port, scientific artifact rewrite or P1 code copy was needed.

## Implemented structure

```text
run_pauli_jw_research_pipeline.py
  → Pauli compose / context / qualification
  → PauliStudyAdapter (explicit study API paths and scope checks)
  → existing same-checkout trotterlib functions
  → synthetic record / versioned SOURCE-result-STOP locators
```

New modules under `src/trottertracks/pauli_jw/` are `contexts.py`, `adapters.py`, `core_stages.py`,
`pipeline.py`, `pipeline_fixture.py` and a side-effect-free package initializer.
The new CLI and `tests/test_pauli_jw_research_pipeline.py` use these same modules.
Only five existing discovery documents receive append-only P2 entries; existing numerical modules and old runners have no edits.
There is no new numerical kernel, scheduler, approval/grant service, all-research schema or alias loader.
Pauli-specific context/record/control code is new because the P1 DF context requires incompatible DF rank/partition fields.
P1's responsibility split is reused; its scientific assumptions and code are not transplanted.

## Connected artificial scopes

|Entry|Method and deliberately narrow scope|Technical connection|
|---|---|---|
|h4-hybrid|saved `p036_q2_r8_K4`, prefix36/q2/r8/K4|existing H4 PR config/law/central trajectory/mean/account|
|h6-hybrid|the existing four fixed signatures: p78/q8/r16,K2; p78/q8/r32,K2; p78/q8/r64,K2; p684/q1/r16,K4|existing completed H6 extension adapter|
|h6-rte|saved prefix0/q8/r64/K2 case|existing pure-RTE event extension|
|h2-index|SOURCE/result/STOP locator entry|plan only; no manufactured H2 provider|

All connected numerical operations are mocks. PF/B0/B0I, other H4/H6 candidates and full H2 wiring are explicitly unconnected.
The generic Pauli kernel already implements deterministic methods; it is retained rather than copied or silently widened here.
H2's script/snapshot boundary was not migrated into this first connection unit. Its available saved evidence remains linked;
this is not a claim that H2 results, code or data do not exist.

## Reused actual APIs

`PauliHamiltonian` receives supplied constant and ordered signed terms through its real constructor signature.
H4 uses `h4_pauli_pr_search.config/law_descriptor/trajectory_seed/central_trajectory/account` and
`h4_pauli_pr_mean._mean_enclosure`. H6 Hybrid uses
`h6_hybrid_fairness_extension.config/law_descriptor/trajectory/mean_target/account_target`.
H6 pure RTE uses `pauli_rte_event_extension.config/law_descriptor/trajectory/h6_mean/account_target`.
Compiler settings and `transpile_and_measure_cost` are the existing APIs, with every compiler field supplied explicitly.
Hadamard wrappers, PauliSignalAdapter, finite RTE and central control remain inside the existing trajectory implementation;
new code neither rebuilds nor double-wraps those circuits.

The generic `fixed_signal_shots` API was inspected but is not substituted for these study-specific accounts:
they preserve their B/u/rho guard and exact-integer conventions. Phase-RPE and DF accounting are not selected.
See [API boundaries](API_AND_STUDY_BOUNDARIES.md) for concrete signatures and missing capabilities.
Seven selected kernel-file comparisons between their scientific SOURCE commits and this base all have identical Git blobs.
That limited metadata check is not full dependency or numerical equivalence certification.

## Technical evidence and use

Actual P2 test run: **109 PASS / 0 FAIL / 1 SKIP**, 110 collected.
Three artificial scopes pass end to end; all four H6 signatures route to their existing callback.
No real state/law/mean/shot/price, RNG, Qiskit build/compile or saved-result reanalysis ran.
The real atomic writer case is skipped because normal legacy package initialization lacks `matplotlib`.
Mock publication tests verify directory/collision behavior separately. [Test scope](TEST_RESULTS.md).

```bash
python scripts/run_pauli_jw_research_pipeline.py --study h4-hybrid
python scripts/run_pauli_jw_research_pipeline.py --study h6-hybrid --mode synthetic
python scripts/run_pauli_jw_research_pipeline.py --study h6-rte --mode synthetic
python scripts/run_pauli_jw_research_pipeline.py --study h2-index
```

Plan/synthetic use the standard library and open no saved scientific input. Default is an explicitly artificial H4 plan.
Scientific/real/resume are refused before input/provider/output or code-metadata access. Import alone starts nothing.
Optional `--output-dir NEW_DIRECTORY` uses the unchanged legacy atomic writer under its original name;
dependency failure retains the original cause, produces an explanatory CLI error and creates no directory.
No dependency is installed and no shared environment is changed.
CLI records the checkout HEAD and clean/dirty code state separately from original SOURCE/result commits;
standalone fixtures identify themselves as unversioned rather than claiming an old commit generated P2 code.

## Evidence and next decision

Architecture `1b649896393a4a0c5f2e3bcf41b2853090ef2053`, adoption handoff `76fec27f45b64879cdf5a86a32772ba78bd0a038` and integrated P1 `2c865c66171fc9d1972b67c1257c57e5684fe26c` were reused as fixed design records.
The new code is not the scientific SOURCE of old results. SOURCE/STOP/cohort/raw/input custody and old runners remain unchanged.
The later [H6 formal handoff](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/CONTINUATION.md)
has an opening completion addendum; its later in-progress text and readiness document are historical snapshots.
No runtime/worker/controller inspection or new independent science verification was performed.

P2 is ready for a **selective adoption review**, not automatic adoption into the summary branch or scientific execution.
The user selects that review/implementation scope next; Codex keeps real providers, P3 and B3 disabled/unstarted.
Normal publication, full remote readback and mandatory STOP complete this task.
