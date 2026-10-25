# H2/H6 Pauli PF/B0/PR conditional resource comparison

Start with `H2_H4_H6_comparison.csv` (PR/D direction) and the technical report
`docs/pauli_h2_h6_resource_comparison_20261025.md`.

- `H2_H6_best_comparison.csv/json`: best B0, full PF, confirmed PR per precision;
  exact prices, bias, B, numerical allowance, N/G, paired SE and source paths.
- `registered_candidate_precision_table.csv`: all 1,675 registered precision rows.
- `H2|H6/candidate_results.json.gz`: exploration and deterministic results. Raw
  per-batch target and law are preserved in the evidence archive.
- `H2|H6/final_results.json`: exploration nominee, its independent confirmation,
  and observed minimum within the frozen confirmation set are separate fields.
- `rare_event_evidence.json`: actual order probabilities and observed counts;
  no unobserved component is assigned zero cost.
- `unresolved_candidates.json`: one retained null row and optimistic diagnostic
  lower bound; this is not a measured G or a population-cost bound.
- `study_plan.json`, per-batch plans, input provenance, `runtime_versions.json`:
  science conditions and source correspondence. No old one-shot/STOP reopened.
- `verification.json`: 2,372 native children exit0/reaped and raw/compressed QPY
  SHA verification. New computation has no implication for old monitor UNKNOWN.
- `data_archives.json`, `H2_evidence.tar.xz.part*`, `H6_evidence.tar.xz.part*`:
  lossless raw/spec/target/QPY/price/native evidence. The raw directories remain
  local and read only after completion; large duplicated files are not staged.
- `SOURCE_history.bundle`: original local SOURCE commits, based on fbfbfd21.
  The result publication may use a GitHub-created commit with different Git
  metadata; this bundle preserves the exact native scientific history.

Restore into a **new empty path** (no scientific calls):

```bash
python3 scripts/restore_pauli_h2_h6_evidence.py   artifacts/pauli_h2_h6_comparison/2026-10-25/data_archives.json   /path/to/new-empty-evidence-directory
```

The script verifies archive parts, raw QPY bytes and original deterministic
gzip containers, then every original member SHA. If another zlib encoder gives
a different gzip stream, use the recorded runtime to reconstruct the container;
the stored QPY bytes and raw SHA remain unchanged.

Native SOURCE history can be imported into this repository, which already has
the base commit, without executing anything:

```bash
git fetch artifacts/pauli_h2_h6_comparison/2026-10-25/SOURCE_history.bundle   HEAD:refs/heads/pauli-h2-h6-native-source-import
```

H4 is a saved fixed-PR baseline, not a PR search. Molecules/states/cohorts are
separate. G assumes rho=0; actual rho, PR population-price tail, global optimality
and formal certificate acceptance remain unproved. Mandatory STOP follows
publication verification; the orchestration scripts are not restart commands.
