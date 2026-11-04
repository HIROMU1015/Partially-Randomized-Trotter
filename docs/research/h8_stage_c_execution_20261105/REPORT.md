# H8長時間・高次PR：Stage C独立確認実験

状態：RUNNING。以下は 2026-11-05T00:36:08.648621+09:00 の起動確認記録であり、確認結果の最終報告ではない。

開始：2026-11-05T00:31:24.562287+09:00。科学実行SOURCE `82fdbdd098e9d3edae01a930437e3d76c1f8bf36`。run_tag `H8_M5_P8_Q1_CONFIRMATION_01`。

Linear H8、隣接1.00 Å、STO-3G、保存tol-only1e-8 DF28、16 system qubit・sector4900、保存state・fragment順序0..27。T=1.6、ε=.0005、P=0。
PRはm5/prefix8/q1/R_outer32/K4、ID `7eb0a3d691b94089969b`。固定N_cos=84916951、N_sin=88339169。
対照は保存B0I/m5/prefix15/q1、ID `1425ae6ed83fd9098b39`、固定G=89291219516568 RZ。新PFのsampling/build/compileは行わない。

CONFIRMATIONの新9 trajectory・両軸18価格。9 workerで起動した。絶対SE目標121655623062.28098 RZ。
Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/target/topologyなし。元の7代表検査・許容値・位相補助を維持する。
新seedの一意性とStage B・除外対象seedとの非衝突、保存入力・SOURCE・compilerの整合を確認した。
読取時点の保存済みtrajectory結果は 0/9。最終平均G・sample variance/SE・比較の達否は未確定。

元Stage A/BのSOURCE・結果・raw/QPY・既存STOPは保全する。探索標本とpoolせず、確認後のn追加・再sampling・候補差替え・自動retry/resumeは行わない。
平均の比較、記述的SE目標、希少orderの観測・unknown費用を分けて報告する。母集団保証を自動認定しない。
private入力・seed実値・event列・QPY・内部pathは公開しない。

[固定計画](../h8_stage_c_preparation_20261104/PLAN.md)、[scope案](../../../configs/h8_long_time/stage_c_proposal_20261104.json)、[保存Stage A/B結果](../h8_long_time_execution_20261103/EXPLORATION_RESULTS_20261104.md)。

次の担当：Codexが9標本の取得・代表検査を確認し、固定PFとの比・差、paired G分散/SE、rare診断、Stage Bとの非pool比較を公開する。固定実験の結果公開後はmandatory STOP。
