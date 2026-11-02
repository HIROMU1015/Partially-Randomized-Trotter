# Pauli B0/B0I prefix/q comparison: saved data

固定linear H4、隣接1.00 Å/STO-3G、Pauli184＋constant、S2/T=.8、central control、Qiskit1.3.0。
初期20＋適応86＝106構成、43 prefix、q1/2/4/8。B0/B0Iは同じ候補で、prefix184はfull PFの端点。
PRは固定prefix27/q8/r4/K4・保存256 trajectoryで、sampling/compile/parameter optimizationを追加していない。

- [技術報告](../../../docs/non_df_h4_pauli_b0_comparison_20261025.md)：条件、候補選択、主要5精度、限界。
- [candidate_membership.csv](candidate_membership.csv)：登録106セルと`NOT_EVALUATED`を区別。
- [candidate_registry.json](candidate_registry.json)：各候補のSOURCE/batch、signal bias/allowance、compile価格、再利用先。
- [candidate_precision_table.csv](candidate_precision_table.csv)：全530行。nullは不適格・未取得で、0や∞ではない。
- [best_by_precision.csv](best_by_precision.csv)：各要求精度の登録集合最小B0/full PFと保存PR。
- [precision_grid.csv](precision_grid.csv)：303点の保存値会計。新しい303実験や連続交点の保証ではない。
- [result_summary.json](result_summary.json)：guard内の順位、適格性数、grid bracket、旧PF会計との区別。
- [analysis_plan.json](analysis_plan.json)、[analysis_provenance.json](analysis_provenance.json)：保存値解析SOURCE/入力/範囲。
- [verification.json](verification.json)、[publication_status.json](publication_status.json)、[STUDY_STOP.json](STUDY_STOP.json)。

## Batch evidence

[batch01_evidence.tar.gz](batch01_evidence.tar.gz)、[batch02_evidence.tar.gz](batch02_evidence.tar.gz)、
[batch03_evidence.tar.gz](batch03_evidence.tar.gz)、[batch04_evidence.tar.gz](batch04_evidence.tar.gz)は、各batchのplan、
run01/SOURCE・target区間・accounting・build metadata・input/output QPY gzip・metrics・child/native記録・stdout/stderrを含む。
[data_archives.json](data_archives.json)のarchive/member SHAで照合できる。展開する場合は空の別directoryへ展開する。
QPY gzipを展開すると、そのbuild/metricsに記録された元QPY SHAと一致する。
rawはlocalにも保持し、GitHubでは巨大な重複JSON/多数の空logをlossless archiveで収録する。
展開・構造読込と、科学runnerの再実行は別である。完了batchを再開しない。

数値はideal targetのbias包含と、ρを仮定した十分shot/Gである。実compiledρ、PR母平均tail、正式certificateは未保証。
PRの元5精度N/Gは引用し、追加gridだけ保存値から会計した。新分子・新state・新PR lawはない。
各精度のα_axis=.025であり、全grid同時95%保証ではない。

全計算は完了しmandatory STOP。保存・公開後に新しい科学計算・重要レビューへ自動移行しない。
