# H8主要6軸実測反映：通常・非相殺限定再会計 run01

2026-10-18 A05説明追補：[通常projectionの固定親accounting、α=.025、証拠分類、実測/model単価とn32 SEの意味](../../../../docs/research/pr_alpha_projection_explanation_20261018.md)。既存のfile_receiptsは元公開commitの記録で、今回の説明追補は別receiptに残す。科学数値は変更しない。

[結果報告](../../../../docs/research/track_a_h8_six_axis_reaccount_run01_results_20261017.md)。SOURCE `aa984d98a7b438dc953a829a24fe3d7edc9d0a6b`、科学dispatch1、3800保存行/40比較、native exit0・wait4回収・最終gate PASS。mandatory STOP。

- `comparisons.json` / `comparison_summary_exact.csv`：旧32/184のbefore/after、PR保存費用・SE、exact削減率、適格/不適格/未解決/欠測。
- `accounting_rows.jsonl.gz` / `accounting_rows_exact.csv`：全3800行の保存値・更新値・保存shot。gzipはnative driver出力byte同一copy。
- `measured_candidate_changes.json`：3候補×10条件×2modeの60行。うち適格34行だけ単価と派生費用を更新。不適格はnullを保持。
- `mode_differences_exact.csv`：通常・非相殺の削減率差。`winner_cost_differences_exact.csv`：保存出力からの表示用exact費用差。
- `saved_output_audit.json` / `cost_provenance_counts.json`：旧32/PR/SE/shot/適格性/非対象単価の不変性とmodel残存。
- `six_axis_provenance.json`：6軸のSOURCE、単価、元scalar hash、native proof。新compile0。
- `authorization_binding.json` / `one_shot_receipt.json`：SOURCE/plan/config/消費済grant/input/profileの公開receipt。実効grant・config・lock・markerの原本は非公開で保持。
- `native_completion.json` / `resource_observations.json` / `final_gate.json`：native exit/wait4/reap、資源・OOM・quota、終了gate。
- `output_receipts.json`：実driverの6出力receipt。`file_receipts.json`：本bundleおよび文書のbytes/SHA（自身は除外）。

通常biasの184最良は5精度中4精度で今回実測単価、1精度は固定model。非相殺は1精度で実測、4精度は固定model。いずれも集合全体の比較には未実測modelが残る。PRは保存n32平均とpaired SEをそのまま使用し、母平均・厳密bias・全候補実測最良は認定しない。新計算・次stageへ自動進行しない。
