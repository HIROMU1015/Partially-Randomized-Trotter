# H8 Phase C run01 結果・終了監査

[資料入口](../../../../docs/research/track_a_h8_phase_c_run01_results_20261017.md)。
SOURCE `931755c4c84aa0751dbde7ddf84cb5187b5172ae`、通常signed biasの保存値1,900行を一回再会計、native exit0・全回収・最終gate PASS。
`comparisons_v1.json` は実driver出力のbyte同一copy、`accounting_rows_v1.jsonl.gz` は1,900出力行の固定mtime gzip、
`comparison_summary_exact_v1.csv` は保存exact値の一覧。`result_summary_v1.json` は候補と費用区分のprojection。
`execution_binding_v1.json` はsource/plan/config/消費済grant/input/profileの結合、`fresh_preflight_v1.json` はfresh gateの公開receipt、
`native_completion_v1.json` はdriver/launcherのnative wait/reapと原本SHA、`saved_completion_audit_v1.json` は保存出力の整合性監査、
`protection_after_v1.json` は自身の実行依存を厳密照合し他worktreeは観測した記録、`cost_provenance_counts_v1.json` は実測/model/PRの区別。
元private control・使用済grant・lockは変更せず保全している。`file_receipts_v1.json` は当bundle自身を除くbytes/SHA。

184集合の最良は全条件model推定である。ε=.05/.0005ではPR点費用が高く、全精度のPR優位は示していない。
S2外挿・S4未測定格子・PR n32の有限標本という限界を保持する。厳密bias・PR母平均・全候補実測最良の保証なし。
追加計算/科学レビュー/次stageへ進まずmandatory STOP。
