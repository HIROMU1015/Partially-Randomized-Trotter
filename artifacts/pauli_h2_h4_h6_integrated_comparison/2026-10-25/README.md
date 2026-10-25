# 保存PF/B0/PR結果の15行統合

[技術報告](../../../docs/pauli_h2_h4_h6_integrated_comparison_20261025.md)。新科学計算・再解析・順位付けは0。
H4だけを最新141構成探索のprimary nomineeへ更新し、旧固定27/8/4/4は主表に使わない。

- `H2_H4_H6_optimized_PR_comparison.csv` / `.json`：同じ15行。数値は元ファイルの文字列を保持し、JSONも文字列。
- `row_provenance.csv`：各行の分子、ε、候補、結果commit、元ファイル、選抜方式。
- `field_provenance.json`：転記列のsource key、行selector、元列名/nested key。
- `source_manifest.json`：10取得元の固定commit、GitHub URL、Git blob SHA、SHA256、snapshot。
- `inputs/`：固定GitHub refから取得した要約・catalog・CSV・報告。旧combined CSVのH4は歴史的baselineで、新主表のH4には使わない。
- `verification.json`：転記・構造照合。`publication_status.json`：公開照合・STOP。

Gは仮想ρ=0、RZ×shot、状態準備費用を除く。PR価格の標本平均に基づく条件付き費用で、実compiledρ、母平均価格tail、大域最適性は未保証。
`PR_C_cos_SE` / `PR_C_sin_SE` はRZ価格のSE、`PR_G_SE` はGのSE。`confirmation_n` はpaired trajectory数で、shot数ではない。
`PR_over_D_exact` はH2/H6の同名列、H4の `PR_over_best_deterministic_exact` を転記した。逆数を算出していない。
空欄は当該正本に表示値がないもの。例えばH4のratio小数を新しく算出せずexact分数を保持する。

`PR_class` はL_D=0で `FULLY_RANDOMIZED_RTE`、0<L_D<Mで `HYBRID_PR`、L_D=Mで全項保持。
MはH2=14、H4=184、H6=918。L_D=0でもconstant phaseを保持する。
H4高精度4行は全ランダム化RTEであり、PR候補集合の低Gをハイブリッド化自体の効果としない。

H4は探索nominee固定後の独立確認。H2/H6は凍結確認集合で観測G最小を選んだ行で、確認後選抜を含む。
SE/保存名目95%区間は選抜後の同時保証や母平均winner保証ではない。
`PR_SOURCE_commits` / `deterministic_SOURCE_commits` は元報告の科学SOURCE、`*_result_commit` は公開結果の固定refである。

再生成は保存snapshotの列転記だけで、科学runnerを起動しない。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/integrate_pauli_h2_h4_h6_saved_results.py
```

成果物の公開・照合後はmandatory STOP。新しい数値解析・科学実行へ自動移行しない。
