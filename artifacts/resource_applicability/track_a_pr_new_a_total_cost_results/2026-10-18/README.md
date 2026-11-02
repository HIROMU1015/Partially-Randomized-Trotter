# 新A_COST保存値総費用比較：完了

[結果と限界](../../../../docs/research/track_a_pr_new_a_total_cost_results_20261018.md)。
SOURCE `8601b921342fc1cc83780a830e1f65bf64fe9e93`、H4/H6/H8各1回、native exit0/reap。

- `new_pr_cost_points.csv`：48新候補×20 endpoint枠=960行。費用/SEは探索n8、shot原表は公開しない。
- `comparisons.csv`：通常/非相殺×5精度×倍率1/10×2決定論集合×3系=120比較枠。
- `completion_summary.json`：3系の終了・SOURCE/binding・件数・原result hash。
- `verification.json`：入力/旧D不変・欠測保持・native waitの作者検査。
- `validation_status_entry.json`：完了状態と未確定事項。
- `files.json`：今回の公開file bytes/SHA（自己参照除外）。

CSVはUTF-8/LF。exact fractionは文字列、`null`は未確定/欠測であり0ではない。
正の削減率はPRの点費用が低い。新n8・旧n32を混ぜず、H4旧n32を独立確認と呼ばない。
H4 legacy DF12、H6 DF19、H8 DF28、1Å/STO-3G/T=.8/P=0。
H6/H8倍率10の新A shot320行、H6非相殺の旧B0+I12候補のsigned成分は未解決。
model/直接実測を分離。母平均・全候補実測最良・厳密bias保証なし。新sampling/compile0、mandatory STOP。
