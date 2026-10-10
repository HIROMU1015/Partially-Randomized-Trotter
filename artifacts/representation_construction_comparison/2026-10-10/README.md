# Limited construction/comparison evidence, 2026-10-10

Report: [representation construction/comparison results](../../../docs/research/representation_construction_comparison_results_20261010.md).
Source commit: `ce99b57166a9f422f151f56fa17ed799cd9909dd`.
Base: `39345830ddfe7c3e2a488c284a0623f489764087`.
Branch: `representation-construction-comparison-20261010`.

`run1/result.json` and `run1/native_ir.json` are raw science outputs. Their source,
input hashes, packages, limits and actual resources are in `run1/run_audit.json`.
374 actual compiled Hadamard wrappers, A18/C48/B2 candidates (B6 q diagnostics).
`precision_resource_rows.csv` displays all 136 saved precision/resource rows;
it is derived from raw JSON and contains no new science conditions.

`saved_evidence_audit.json` verifies 176 commit source/input blobs and all 374
native IRs using a separate NumPy/SciPy implementation (no Qiskit or project
imports). Native counts/depth, absolute phases, controlled branches, stored
event sequences, finite corrected means, biases, normalizations, shot budgets
and paired X/Y cost statistics are checked. Three mutation cases are rejected.
It is a local saved-data audit, not external scientific reproduction or CI.

From a checkout containing this verifier and evidence, saved-only verification:

```bash
python scripts/verify_representation_construction_comparison.py \
  --run artifacts/representation_construction_comparison/2026-10-10/run1 \
  --output /tmp/representation-construction-saved-audit-new.json
```

The output must not already exist. Dependencies used are recorded in
`environment_requirements.txt`; the existing isolated environment was reused.
To reproduce the scientific batch, use the frozen source commit rather than a
moving branch and a fresh output namespace. No scientific replay was performed
after run1. Current mandatory STOP does not authorize another science stage.

`tests_v1_failure.log` preserves the pre-freeze two-failure keyword mismatch;
`tests_v2.log` and `tests_v3.log` preserve the subsequent 67/70 passes.
`run1_console.log` includes the successful run and outer `/usr/bin/time -v`.
`independent_algebra_verification.json` records semantic reproduction of the
user ZIP rational checks. Original ZIP entries and manifest remain under
`docs/research/representation_construction_inputs/algebra_checks/`.
`provenance.json` binds original review/ZIP identity and copied inputs;
`protected_state_audit.json` records root and prior series preservation.

`remote_retrieval_receipt.json` is saved in a publication child commit after a
fresh GitHub fetch of the result commit, without local Git object alternates.
It certifies retrieval/identity, not rerunning the scientific experiments.

Terminal status: `LIMITED_CONSTRUCTION_COMPARISON_COMPLETE_AWAITING_GPT_REVIEW`.
`mandatory_stop=true`, `next_stage_authorized=false`,
`central_hypothesis_adopted=null`. Quantum shots, molecular loads, GPU and
ground-state solves are zero. No general novelty, molecular compression,
fault-tolerant or final energy/QPE/RPE resource claim is made.
