# C4 Q13 limited execution: attempt01

`C4_LIMITED_CORRECTNESS_PASS_RESOURCES_RECORDED_MANDATORY_STOP`.
One execution, no interruption or retry. Frozen science source:
`b546c4e4f5d6645fd198b9ec3f85e3cd644ac4e9`.
[Contract, methods, results and limits](../../../../docs/research/research_c_rustiq_execution_20261024.md).

The saved F1 synthetic fixture has four modes/two particles, L_D=1, two residual
fragments, T=1/5, q=2, delta=1/10 (one point), r=2, h=1/20, K=2. Molecular
geometry/AO basis/DF rank do not apply. Only SS motifs2/3/4 and PP motif0 were
compared through B0/B1/R1 at the unchanged Qiskit1.3.0 compiler profile.

- [summary.json](summary.json): completed call counts and maximum errors.
- [metrics.csv](metrics.csv), [comparison.json](comparison.json): passing full
  measured wrappers; eight designed case/axis records per route. Unweighted
  totals are not a law expectation or total-round resource advantage.
- [provenance.json](provenance.json), [approval.json](approval.json),
  [preparation_contract.json](preparation_contract.json): source/input/API/lock
  identities, this user's new approval and technical-retry amendment.
- `saved_inputs/`: exact seven synthetic saved input files, including fixed law
  binary64 payloads and requirements lock; old originals remain unchanged.
- `circuits/`: 92 QPY files; `operators/`: 68 NPY files; `references/`: 28
  gzip-compressed JSON bundles at 90/128 digits.
- `journals/`, `reports/`, [console.log](console.log), [resources.json](resources.json):
  312 balanced registered requests, absolute gates, ordered typed occurrences,
  phase records, cold/warm timing and process resource configuration.
- [classical_costs_by_route.json](classical_costs_by_route.json): saved-only
  reconciliation; `python summarize_saved_results.py --check` reproduces the
  integer cost comparison without Qiskit, circuit construction or simulation.
- [OUTPUT_HASH_MANIFEST.json](OUTPUT_HASH_MANIFEST.json): exact bytes and SHA256
  of every packet file except the manifest itself.

[STOP.json](STOP.json) records completion. Further experiments, law-weighted
averages, an identical-Pauli angle-merge control, certified shots and general
algorithm advantage are outside this execution's evidence. This is committed
local evidence, not CI or independent external reproduction.
