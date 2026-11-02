# H4 PR母平均回路価格：限定scalar実行結果

[結果報告](../../../docs/non_df_h4_pr_mean_cost_scalar_execution_20261025.md)。固定SOURCE `fa3337b57e7d1e6ba5296317770a4f49a0cbaaf1` を変更せず、承認されたscalar batchを1回実行した。native exit 0、両軸で十分条件が分離した。

- [scalar_result.json](scalar_result.json)：固定入力、6 SOURCE/input SHA、Pauli upper、DF lower・共通ln/eのexact有理数端点、4種類のstatus。
- [execution.json](execution.json)：実行コマンド、UTC時刻、runtime、native returncode、stdout/stderr。再実行用の指示ではない。
- [verification.json](verification.json)：SOURCE保全、準備済み人工12検査の再利用、今回の計算範囲とSTOP。独立科学レビューではない。
- [publication_status.json](publication_status.json)：専用branchの公開・SHA照合。

価格はstate準備を除くfull-wrapper compiled RZ数。cosineはPauli upper 12,391、sineは12,392。DFの統計的母平均下界の数値包含は両軸とも `[19741.473302966152, 19741.473302966153]` RZ（表示用に外向き丸め）。この狭い区間は**下界値の数値包含**であり、DF母平均そのものの信頼区間ではない。

95%同時coverageは固定effective lawのIID仮定とunion boundに依存する。Pauli countは正常returnするSOURCEモデルの補題で、全allowed入力の実compiler正常終了とRust SOURCE／wheel対応は未証明のまま。数値分離を無条件の全law保証、G順位、ρ保証、正式certificateへ昇格しない。旧非実行仕様・旧STOPも変更しない。

**科学計算は完了・mandatory STOP。追加compile／sampling／N/G／law-weightedへ自動移行しない。**
