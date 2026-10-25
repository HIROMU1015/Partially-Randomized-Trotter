# Hybrid PR / fully randomized RTE：固定3分子の条件付き比較

[技術報告](../../../docs/pauli_hybrid_rte_comparison_20261025.md)。準備commit `25d248a1f2ed88de62b0e621c2d1cea8532804c4`。
科学driver SOURCE `b19ed83c6b4b64821965fa0b756cd2691e34eda8`。
新SOURCE/入力/nomineeを固定し、既存run/STOPを再開していない。新しい承認証明・grantは作成していない。

- `comparison_by_precision.csv`：15条件。探索と確認の方式別min、Hybrid/RTE比、確認n/SE、決定論的保存基準。
- `comparison_by_precision.json`：選択行・exact tie・未解決ID。G欠測は0/∞にしない。
- `cost_breakdown.csv`：選択候補のB、signed bias、allowance、軸別N/RZ、G。元Nを再計算しない。
- `initial_comparison.json`：新sampling前の保存値比較。`nominations.json`：確認前固定ID/精度/保存行。
- `new_candidate_accounting.csv` / `new_price_statistics.json`：新確認候補の価格と保存NによるG。paired統計を保持する。
- `historical_baseline_separate.json`：H4旧p027/q8/r4/K4。探索/確認へpoolしない。
- `SOURCE.json` / `active_scope.json` / `runtime.json` / `seed_schedule_verification.json`：科学SOURCE、固定範囲、同一入力、未使用seed。
- `archive_restore_verification.json`：3 archive・1020元ファイルの復元SHA照合PASS。`engine_native_exit_observation.json`：元engineの直接取得exit0。
- `comparison_scope_status.csv`：有限部分集合の経験的分類と未解決候補を明示。
- `completion_verification.json`：保存値・QPY SHA/RZ/CX/wire・native exit/reap照合。
- `STUDY_STOP.json`：科学実行のmandatory STOP。`engine.*.log`：今回engineの標準出力/エラー。
- `data_archives.json` / `*_evidence.tar.xz.part*`：新raw QPY/JSON、batch plan、入力コピー、native記録を同じ旧QPY-aware形式で可逆保存。

Gは仮想ρ=0、状態準備を除くRZ×shot。実compiledρ、PR全law残差、母平均価格tail、大域最適性は未保証。
方式別の確認minは事後選抜であり、元H4 primary nomineeは別列に保持する。
`*_G_SE` はGのSE、内訳の`C_cos_SE/C_sin_SE`は価格SE。両者を混同しない。
名目2SEのnear-tie区分・逐次延長は、選抜後の厳密な同時保証や母平均winner保証ではない。
H6にはN/GがSAFE_INTEGER_LIMIT等の行があり、minは解決済み有限G部分集合の範囲である。

可逆復元は既存のgeneric restorerを再利用する。復元先は未作成の専用directoryとし、旧runやSTOPへ上書きしない。

```bash
python3 scripts/restore_pauli_h2_h6_evidence.py \
  artifacts/pauli_hybrid_rte_comparison/2026-10-25/data_archives.json \
  /tmp/pauli-hybrid-rte-evidence-restored
```

復元先のH4/H6/stagesは保存証拠の閲覧用であり、科学runの再開指示ではない。
今回の実行は `scripts/run_pauli_hybrid_rte_comparison.py`、結果照合は `scripts/check_pauli_hybrid_rte_results.py`。
完了したscientific runを再実行しない。結果公開・local/remote SHA照合後はmandatory STOP。
