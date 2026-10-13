# C1 pilot v1：GPT科学判断の引継ぎ

2026-10-13 JST。実行・独立監査は完了。GPT科学レビューの開始は別の明示承認待ち、Mandatory STOP。
[結果・scope・claim限界](qc_c1_macrocycle_pilot_v1_results_20261013.md)、[固定事前登録](qc_c1_macrocycle_preregistration_v1_20261013.md)、
[技術設計](qc_c1_macrocycle_technical_design_v1_20261013.md)、[完了選定レビュー](qc_four_algorithm_candidates_scientific_selection_review_2026-10-13.md)。

不可欠な証拠：固定source `ddaa05d2b51d575c3cfea1194bbfdffa182f9636`、入力・source/runtime manifest、実行認可、
一次4caseと936 phase records、selector seal、32比較、independent audit、実行/profile/STOP。
[保存証拠集約・SHA索引](../../artifacts/qc_c1_macrocycle_execution_v1/2026-10-13/saved_results_summary.json)、
[実行監査](../../artifacts/qc_c1_macrocycle_execution_v1/2026-10-13/execution_audit.json)から辿る。
新sourceや文書だけを理由に旧選定レビューを反復する依頼ではなく、一回結果を受けた重要科学判断への引継ぎである。

固定4非分子2qubit toy・39方法・6h・target10^-5/10^-6 model units・controlled価格3/6。
940 eig calls、保存監査936unitary/8064cost rows PASS、追加eig0/retry0。
通常state16比較でempirical/joint/box全て5%改善0、bad state16比較はoverlap条件未達。
真の3主要誤差係数は全4caseで正、joint coefficient-only hは6/32valid、26/32null。
known-composition生成器同値性を維持し、全ground-energy量子資源費用はUNKNOWN。

次レビューの論点：

1. 同符号・既知composition・強い細分化/S4/C2対照で優位0という結果から、現C1系列を区切るか。
2. この狭いfinite-input結果と、全PF/分子での不能性をどう区別するか。
3. 異符号など未観測機構を狙う再設計に具体的な情報価値があるか。成功例が出るまでの同型scanを認めない。
4. 次点C4などへ移る場合、別の採択判断・固定scope・実行認可が何に必要か。

Codexは研究の正式継続/終了/新規性を採択していない。review開始承認前にこれらを深く比較・選定しない。
GitHub commit/pushとremote取得確認を完了してから公開証拠の固定URLを追記する。
