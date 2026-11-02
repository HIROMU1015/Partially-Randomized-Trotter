# PR/PFのC・N・G：保存値会計と作図

[全成果物・条件・図](progress/track_a/figures/pr_pf_analytic_vs_circuit_CNG_20261022/README.md)。固定結果a590214から新しい専用branchを作り、SOURCE bb5251fcで候補×精度ごと40単位の保存値会計をone-shot実行した。旧SOURCE、raw、grant、ledger、marker、STOP、図、スライドは変更しない。

CとG00/G01は20条件で対応する。H2/H4の10条件は実施済み技術検査PASS範囲、H6/H8の10条件はunweighted-G補助物理検証不合格の留保付き。N_circuit・N比・G10/G11・N変更寄与は未取得null。GはN固定・Cのみ置換した比較であり、独立Nの差が0と確認したものではない。

対象はlinear H2/H4/H6/H8、隣接1 Å、STO-3G、tol-only DF1e-8・rank3/10/19/28、generation順、T=.8、外側S2、旧最良候補固定、P=0、signed bias・経験的allowance倍率1、α_axis=.025、両Hadamard軸、native DF/symmetric-directional/original_plus_tail_once/reference_rotated=false。Qiskit1.3.0・Nature0.7.2・rz/sx/x/cx・opt1・seed17・topologyなし。各L_D/q/r/K・δ=.8/q・入力hash・cohortはpacketに対応する。原論文のToffoli資源の直接再現や全RPE最終費用ではない。

Nの既存定義は補正付き±1測定平均に対するHoeffding十分条件。biasは古典的な平均信号／参照評価、normalizationはfinite-RTE法則、uは経験的。compiled平均信号から独立したN評価は保存されていない。全法則での理想同値性と同じ推定器・bias・u・B・αを仮定すれば同じ式が同じNを返すという数学的帰結と、数値としての独立比較を区別した。新推定器の採択や実行は行わない。

純粋な会計libraryは `src/trotterlib/analytic_circuit_cng.py`、one-shot runnerは `scripts/run_analytic_circuit_cng.py`、人工testは `tests/test_analytic_circuit_cng.py`。表示更新は `scripts/plot_analytic_circuit_cng.py`、別処理の有理数／Decimal・共分散・欠測・SOURCE・保全／図形式／リンク検査は `scripts/verify_analytic_circuit_cng.py`。保存実行記録は `artifacts/resource_applicability/pr_pf_analytic_vs_circuit_CNG/2026-10-22/`。

40単位の科学的保存値会計を再実行しない。表示修正だけは保存結果から行う。既存研究原本保全の明示指示を優先し、研究概要・目的・スライドへは追記せず、この検証文書・索引・新日付ノート・証拠台帳へ記録する。

新signal、shot評価、sampling、compileは0。技術テストは4人工件、保存値の会計検査は40 method side・80軸、PNG/SVG/PDF30組。旧1131ファイルのhash不変。PRの名目MC区間と希少event・母平均の保証限界、H2 n8と新MCの分離を保持する。主担当のlocal技術確認で、GPT独立詳細科学レビューではない。結果保存・ローカルcommit後mandatory STOP、push・公開・merge・PR作成なし。
