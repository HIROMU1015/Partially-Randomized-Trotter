# H6 Pauli-JW Hybrid PR対RTE：比較条件の追加検証

[技術報告](../../../docs/h6_rte_fairness_20261025.md)。科学SOURCE `54cd027431da92736e5f8b9a14c1a0a88760288a`、基点 `90c1c60820d52ca7ddcd4fb5315e9d21bb4a86c3`。

- `registered_RTE_candidates.csv`：L0、q1/2/4/8、r2/4/8、K2/4の24構成。旧7を再利用、新17のみ実行。
- `expanded_RTE_inventory.csv`：5精度・探索/確認を分けたN/B/bias/u/価格/SE/G。Nはexactな十進整数文字列。nullは未取得/未確定であり0や∞ではない。
- `comparison_by_precision.csv/json`：matched、探索family min、確認family min、確認前nominee、旧r4比較、保存最良決定論的PF/B0を別fieldで保持。
- `matched_comparison.csv`：同一q/r/K対照。`selected_cost_breakdown.csv`：選択候補の内訳。
- `new_candidate_results.json` / `old_rte_accounting_extended.json`：今回の数値会計詳細。旧結果を変更しない新しいview。巨大整数JSONは任意精度対応readerで読む。
- `integer_limit_investigation.json`：旧SAFE_INTEGER_LIMIT/FIXED_GUARD_FAILEDの実際の判定と新会計の対応。旧null8行を回収。
- `unresolved_candidates.csv`：最厳精度の数値guard未確定2行と、同じ固定観測価格だけに対するG下界。母平均価格の下界ではない。
- `SOURCE.json` / `analysis_SOURCE.json` / `study_plan.json` / `runtime.json`：入力・code・数値条件・科学範囲。
- `batch02_decision.json` / `batch03_decision.json` / `confirmation_nominees.json`：追加候補の選択理由と確認前固定ID。
- `verification.json`：QPY SHA/数/wire、seed、exact G/shot十分条件、旧入力保全、native241 child exit0/reap。`engine_native_exit_observations.json`：元engine4件の直接取得exit0。
- `result_summary.json` / `STUDY_STOP.json`：今回の完了・科学STOP。新target17作用、新224 trajectory（探索128＋確認96）、両軸448 compile。Hamiltonian/state/reference生成0。
- `data_archives.json` / `*_evidence.tar.xz.part*`：新model/referenceコピー・個別law/spec・target包含区間・raw QPY/JSON・nativeログ/plan/STOPを保全。元gzip bytesをidentity encodingで保存し、既存restorerを流用。
- `archive_restore_verification.json`：2 archive・1664元ファイルの復元SHA照合PASS。元rawは保持する。

復元先は未作成の新directoryを使用する。これは証拠の閲覧であり、科学runの再開指示ではない。

```bash
python3 scripts/restore_pauli_h2_h6_evidence.py \
  artifacts/h6_rte_fairness/2026-10-25/data_archives.json \
  /tmp/h6-rte-fairness-evidence-restored
```

復元された `H6/batch*/cases/*/target.json` にideal law-mean signal（corrected mean）、bias、数値包含区間とnormがある。対応する `spec.json` にfinite event law/CDF/normalizationを保持する。`prices/*/*/result.json` とQPYは両軸paired価格、`confirmations/cohort_1` は独立確認/固定延長判定。

Gは仮想ρ=0・有限標本の平均RZ×保存/新N。実compiledρ、PR全law残差、母平均費用tail、全連続パラメータの最適性、正式certificateは未保証。探索と確認をpoolせず、確認集合minは事後選抜を明示する。科学実行は再開しない。公開・local/remote SHA照合後もmandatory STOP。
