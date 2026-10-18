# PR α B-2 run01

[結果・固定条件・限界](../../../../docs/research/pr_alpha_sensitivity_run01_results_20261018.md)。2026-10-18 JST、一回実行、総α=.01/.05/.10、各軸へ半分、SOURCE93145a4b…。`COMPLETE_WITH_INCONCLUSIVE_MANDATORY_STOP`、driver/launcher exit0。

- `result.json.gz`：private result.jsonの無損失gzip。11,160会計行、66比較、66切替記録。raw signal・reference・private grant原本は含まない。
- `comparisons.json` / `comparisons.csv`：最良候補、exact有理数費用・比・削減率、paired費用SE、実測/model/欠測の分類。
- `switches.json`：基準α=.05との比較。未確定を維持し、null winnerの不変を確定順位の検証数へ加えない。
- `summary.json`：3,660基準再現、既知60入力欠測/180出力欠測、確定34順位比較で変更0、削減率変化最大.00471244パーセントポイント。
- `execution_binding.json`：SOURCE/runtime・承認scope・private原本のhash receiptのみ。
- `terminal.json` / `native_wait.json` / `launcher_wait_receipt.json`：一回実行とnative終了・wait完了。
- `verification.json` / `preservation_audit.json`：保存出力の整合性と旧70 fileの保全。
- `artifact_receipt.json`：公開fileのbytes/SHA256。receipt自身は対象外。

[回収checker](../../../../scripts/collect_pr_alpha_sensitivity_run01.py)は保存値の照合とexportのみ。科学runnerやshot/二項関数を呼ばない。入力引数はprivate run・plan・未使用出力directory。実行済み科学計算を繰り返すためのscriptではない。

全会計EMPIRICAL、証明書未検証、母平均winner保証なし。PR n32 SEは測定αの証拠ではない。単一signalの会計を複数signal・選択後・RPE保証へ拡張しない。新signal/sampling/compile/fitなし、自動retryなし、mandatory STOP。
