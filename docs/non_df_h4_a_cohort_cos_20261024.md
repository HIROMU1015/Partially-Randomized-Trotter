# H4 Pauli-JW PR central-control：A_COHORT_COS

2026-10-24 JST。添付1c9db243の明示承認に基づき、元の保存256 trajectoryのうちindex2〜255だけを新規評価する。既存D2 index0（cosine1145 RZ）とA_GATE index1（1129 RZ）はSOURCE・固定compiler・入力/QPY SHA・native証拠が適合し、新compileなしで再利用できた。基点5eaf32d2f01e95f514634bbbdefd43e971da5d16はGitHub取得済み。現時点はSOURCE固定前の技術記録で、科学結果は追記する。

Linear H4/隣接1.00 Å/STO-3G、system8＋ancilla8、Pauli184非identity項・PR prefix27、S2/T.8/q8/r4/K4、保存law・eventを維持。Qiskit1.3.0、rz/sx/x/cx、opt1/seed17、backend/coupling/layout/routingなし、cosine Hadamard wrapper、状態準備費用除外。D2 central-control、A_GATE body-only接続、元D1 append/compiler、D2並列batch・grant/CallBudget/native witnessを再利用する。新builder/compilerは実装していない。

[接続](../src/trotterlib/h4_a_cohort.py)、[runner](../scripts/run_h4_a_cohort_cos.py)、[人工12件](../tests/test_h4_a_cohort.py)、[証拠入口](../artifacts/non_df_h4_a_cohort_cos/2026-10-24/README.md)。新人工12 PASS。A_GATE12＋D2/比較再設計20の保存意味対応証拠も保全して再利用し、今回再実行したとは扱わない。主agent自身の検査で、別担当による独立reviewではない。

call最大はbody254・central254・cosine wrapper254・費用top-level compile254・統計1固定batch。compileは各taskの起動前に非返還予約し、失敗も消費済み。index0/1・genericの再compile、sampling/RNG、sine新compile、N/G、残差診断、D3、正式certificate受理は0。SOURCE対応失敗はそのtaskのcompile前にSTOP。失敗waveより後は投入せず、既投入taskのnative終了を回収し、欠測を0で補完しない。

並列度はresource_plan.jsonの現実の空きCPU・メモリ観測に基づく。独立compiler processを十分に使い、各processの小行列処理はBLAS/OpenMP1 threadとして多重thread生成を避ける。旧runのthread/worker制限を恒常上限として流用せず、共有設定・他processは変更しない。SOURCE・全入力/runtime/旧保全対象は実行前後に照合し、各taskでも使用code/model/自分のjoinを照合する。inspectorは各task一度読込む。

254/254の成功と既存2件の適合確認後だけ、固定256本の価格をtrajectory IDでpairedにして1 batch集計する。既存_statisticsのnominal Student-t95・sample ddof1とnumpy.linear分位点を使用。保存cohortのexact平均と、implemented law母平均・未観測order4 tail・compiler残差保証を区別する。PauliとDFは異なるlawでありindexでpairedにしない。Pauli PF15028、旧generic Pauli PR平均9321.1015625、DF PF135355/PR平均23678.44140625は保存値として引用する。

旧Pauli-PR historical hash FAIL/原因UNKNOWN、元H4 native UNKNOWN、旧N/G null、C/L COMBINATION_BLOCKED、旧compiler MISSING_EVIDENCE・正式certificate未受理を維持する。研究運用§8.4/10.1/10.2適用、同ファイルに存在しない§10.3は仮定しない。

専用branch `codex/non-df-h4-a-cohort-cos-execution-20261024` にSOURCEをcommitしてからsealと新one-shot grantを発行する。実行・保存・終了監査・commit/push後mandatory STOP。次の担当はユーザー：A_AXESや条件付きC/N/Gへ進むか別承認で判断。Codexは自動移行しない。
