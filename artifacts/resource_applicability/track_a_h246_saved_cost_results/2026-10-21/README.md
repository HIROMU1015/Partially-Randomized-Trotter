# H2/H4/H6高次PF：保存値統計・総費用

固定取得結果H2 a05f4ed19、H4/H6 605ca07a6。解析SOURCE 9a77d2b0a。旧signal・cohort・compilerを変更せず、新規sampling/signal/simulation/回路生成/compileは0。

- `evaluation.json`：100候補、1000候補/精度/bias記録、30まとめ。科学実行が保存した値を転記した集約JSON。trajectory原本・seed一覧は含まない。
- `candidate_precision_costs_1000.csv`：通常signed/非相殺を分けた全候補表。shot、bias、u、normalization、費用、SE、exact有理数meanを対応させる。
- `formula_comparison_30.csv`：3系×5精度×2biasのS2/Yoshida4/m5の探索点最良。
- `cost_decomposition_60.csv`：Yoshida4/m5対S2の総費用比、shot総数比、有効単価比。
- `unit_statistics_36.csv`：PR36群の軸別単価平均/不偏分散/SE/共分散。
- `observed_strata_144.csv`：36群×S0/S2_single/S2_multi/S4。0観測層の費用・分散はnull。
- `independent_numerical_checks.json`：有理数/中心化float、凍結shot/Decimal、正確な費用分解の照合記録。libmのsqrt/exp/log共有は明示。
- `policy_adoption.json`、`grant_and_scope.json`、`input_provenance.json`、`end_audit.json`：新しい限定政策、消費済みone-shot、入力来歴と終了証拠の公開投影。
- `structure_checks.json`：CSV/JSON対応・null・元SOURCE/結果・旧worktree保全・リンク・公開範囲の最終検査。

CSVの`null`は欠測・未評価・対象外を表し、0ではない。大きな総費用の正確なmeanはnumerator/denominatorを使う。通常signedと非相殺、標本SEと母平均保証、PR比較とm5決定論部分集合の比較を区別する。旧強化S2/S4対照を置換していない。AD03保留。local commit後mandatory STOP、pushは別承認。
