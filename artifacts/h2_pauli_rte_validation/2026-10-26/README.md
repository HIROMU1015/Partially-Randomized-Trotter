# H2 pure RTE — confirmations and limited event extension

2026-10-26。科学SOURCE `bf5f2339f723f4edb7bf91a8ca88f33251bbef57`。
[技術報告](../../../docs/h2_pauli_rte_validation_20261026.md)を入口とする。
新しい値は全てH2専用scopeで、旧H2/H4/H6の成果・STOPは上書きしない。

## データ

- `old_candidate_confirmation_map.csv`：旧純RTE13構成と旧/新confirmation nの対応。
- `candidate_precision_table.csv`：Phase A保存target/N＋新確認価格、Phase B新target＋探索価格。q/r/K/s、cohort、n、B/bias/u、軸別N/C/SE、G/paired SE、rho guardを収録。
- `event_cost_trends.csv`：旧s2〜32と新s64/128を出典・cohort別に収録。値のない項目は未取得のまま。
- `comparison_by_precision.csv`／`comparison_details.json`：旧確認済みRTE、Phase A最小、Phase B探索/確認、更新後確認集合minと保存Hybridを分離。
- `candidate_Hybrid_ratios.csv`／`selected_cost_breakdown.csv`：全新cohort行の保存Hybrid比と、選択されたHybrid/旧RTE/更新RTEのB/N/C内訳。完成保存Gからの会計表示のみ。
- `candidate_results.json`：新cohortの詳細会計。Phase Aは元Nをexactに再利用し、shot再計算なし。任意精度整数/有理数対応readerで読む。
- `study_plan.json`：科学実行前に固定した14候補universe（A8、B初期3/条件付き3）、有限call上限、96→192→384確認規則とSTOP。
- `Phase_B_confirmation_nomination.json`：Bは競合せず、追加確認/条件付き候補なしという保存判断。

固定linear H2/隣接1Å/STO-3G、保存Pauli14項＋constant、S2/T=.8、純RTE L0、central/Qiskit1.3 opt1 seed17、状態準備を除くRZ×shot、仮想rho=0。
実compiled rho・母平均費用tail・大域最適性は未保証。
全5精度は価格標本を共有し、独立な5実験ではない。探索と確認は混合しない。
確認集合minは事後選抜を含み、各候補の母平均最適性やtail保証ではない。

## 図

`figures/H2_G_vs_s.png/svg/pdf`：5精度のlog-s/log-G、保存Hybrid水平線。qとK、探索/確認を分離し、線は視覚的ガイド、誤差棒は名目2SE。
`figures/H2_B_N_C_vs_s.png/svg/pdf`：B、cosine C、ε=.0005のN_total。Cは確認価格があれば確認、なければ探索を使う。元CSVのcohort/nを必ず併記する。
同じsの異なるq/rが同じ実装価格になるとは仮定しない。

## 保存と復元

`SOURCE.json` は新H2 runnerと再利用する実コード/保存inputのSHA。
`analysis_SOURCE.json` は最終整理コードとsnapshotのSHA。
`prior_inventory.json` は旧H2の引用行、`prior_evidence_selection.json` は既存archiveから復元して照合した元member SHA。
`engine_completion_observations.json`／`verification.json` は実native終了・回収・QPY・会計照合。人工検証は技術検査で、科学的優位の保証とは区別する。

`data_archives.json` と `*_evidence.tar.xz.part*` を既存restorerで復元する。
`H2`には新case/spec/law/target/両軸input+compiled QPY、`stages`と`confirmations`には先行固定plan/nativeログ/exit/reap/延長判断/STOP、`prior_evidence`には旧13候補/Hybridの選択復元証拠を収録。
元rawを削除・変更せず、Gitでは可逆archiveを公開する。

```bash
python3 scripts/restore_pauli_h2_h6_evidence.py \
  artifacts/h2_pauli_rte_validation/2026-10-26/data_archives.json \
  /tmp/h2-pauli-rte-validation-restored
```

出力先は未存在directory。これはbytes復元だけで科学再実行ではない。
`archive_restore_verification.json` は全memberの元bytes復元SHA検査。
`STUDY_STOP.json` 後の科学run再開は禁止。次はユーザー/GPTが研究上の扱いを判断する。
