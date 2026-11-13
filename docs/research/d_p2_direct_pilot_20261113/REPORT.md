# P2直接paired残差pilot

2026-11-13 JST。ユーザーが明示承認した有限範囲を、基点
`5a67cea7705e80ff295306606611d774724ae2e2`から既存Dブランチで実行する。
準備段階：専用runner・人工16 test PASS、[PLAN](PLAN.json)固定。
科学結果は実行完了後にここへ記載する。

対象は固定Linear H4/1.00 Å/STO-3G・保存tol-only DF10、S2/L_D1/K2/h=.1/M12、
元exactκ/H/state/snapshot、symmetric-directional full Hadamard X/Y。
controlled ZZ=`CONTROLLED_ZZ_CX_CRZ_CX_V1`、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17。
replayはP1 run02 m0/index0とm6/index0、両軸4 compileまで。
全replay一致後だけfresh cohort `D_P2_DIRECT_PILOT_V1`、m0/m6各index0..3、16 compileまで。
追加path/round、retry、forced rare、fit/外挿、全13round平均価格認定、N/RMSE/G、8次PFは実行しない。

元C1 context・sampler/compose、compiled_price_fast、paired_residuals/cohort_summaryを再利用する。
replayは保存component列から元event関数で再構成し、各eventのJSON表現とdigestを照合する。
乱数による再samplingはしない。private原本は読み取り専用。
replay2 process、fresh8 processを各phase内でpathごとに並列化する。
各pathのX/Yは個別build/compile、compilerは既存どおりnested poolを使わない。
一時的な共有CPU競合を理由に正常workerを止めず、共有設定・他processは変更しない。

科学実行SOURCEはこのrunner/PLANを含むcommitで固定する。
今回のpilotは平均・ばらつきと将来n/負荷の計画材料であり、2%/95%達成や全law資格を認定しない。
不一致または実行不能は部分記録を保全してSTOPし、無断retryしない。
