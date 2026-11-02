# H8 resource-map結果：独立32確認後の費用比較

2026-10-14。SOURCE `c31ef424175c02379eafebe8fc6c7548c1d9aabe`、科学seed source `cb8a56c6b2a772827275f9200b02638030f33f72`。branch `track-a-h8-resource-map-resume02-20261014`。

H8で探索592 group、事前選定6構成それぞれ独立32 trajectory（192 group）が完了した。独立確認後の点最良PRは、各要求精度でfull PFより測定込みRZ総費用が54.29–65.76%低く、prefix-only PF B0と比較しても39.48–56.12%低かった。母平均上界や同時winnerの厳密保証ではなく、固定ensemble signal・経験的数値allowance・保存compiler・状態準備費用P=0に条件付けた点推定である。

## 対象と費用定義

linear H8/1.00 Å/STO-3G、16 spin modes・電子alpha4/beta4・sector dimension4900、tol-only DF1e-8/cutoff0/signed generation order/weighted Hermitian projection・actual rank28。保存Hamiltonian/state/DFを変更せず、単一finite time T=.8のcomplex signalを評価した。PF係数Cのdelta窓fit、chemical accuracy、QPE/RPE energy精度の認定とは別。

B0=prefix-only deterministic S2、B1=全rankのdeterministic S2/global Yoshida S4、B2=partial S2＋finite RTE。B0/B2 prefix7/14/21、q1/2/4/8、B2 r1/2/4、K2/4、92 signal候補を固定。symmetric_directional cosine/sine full Hadamard wrapperを使用し、Qiskit1.3.0/basis rz,sx,x,cx/optimization1/seed17/topologyなし、compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b` を維持した。

G_RZ = N_real × mean(RZ_cosine) + N_imag × mean(RZ_sine)。回路RZ/CX/depth/sizeは実compileから取得し、Nはbias・normalization B・経験的u・axis allocation・alpha_axis=.025を含む既存Hoeffding会計による必要shot見積り。ハードウェアshotsを実行した値ではない。状態準備・readout mitigation・fault-tolerant overheadは含めず、共通状態準備はG(P)=G(0)+N_total Pとして別感度にする。

## 精度別の独立確認結果

表のPRは全てprefix7・r4・S2。確認前に各精度の探索点上位2の和集合6構成を固定し、その各32本の結果の中で点最良を報告する。確認後の点選択を新たな事前選定や母平均winner証明と呼ばない。B1 PFは登録q1/2/4/8・S2/S4の精度適格候補中の点最良、B0はprefix7/14/21の精度適格候補中の点最良。

| epsilon_sig | PR構成（p7/r4） | full PF構成 | PR G_RZ | full PF G_RZ | full PF比削減 | B0比削減 |
|---|---|---|---:|---:|---:|---:|
| 0.05 | q2_2nd_r4_K2 | q1_2nd | 1.369e+09 | 2.99512e+09 | 54.29% | 39.48% |
| 0.01 | q4_2nd_r4_K4 | q4_2nd | 6.07921e+10 | 1.7062e+11 | 64.37% | 52.72% |
| 0.005 | q4_2nd_r4_K4 | q4_2nd | 2.96433e+11 | 8.32295e+11 | 64.38% | 52.73% |
| 0.001 | q8_2nd_r4_K4 | q4_4th | 1.536e+13 | 4.48582e+13 | 65.76% | 56.12% |

点最良PRのsignal absolute biasは、epsilon .05/.01/.005/.001に対してそれぞれ3.547069e-3/8.771512e-4/8.771512e-4/2.186978e-4。適格判定はreal/imagのheadroom・経験的uを含む既存会計であり、absolute biasだけで判定していない。32 trajectoryは平均回路費用の独立確認であり、新しいsignal精度証明や実機測定ではない。

探索8本ではq2K2/q4K2/q4K2/q8K2が点最良、独立32本ではq2K2/q4K4/q4K4/q8K4だった。q4/q8のK2とK4の費用差は小さく、有限sample・事後点選択の影響を含む。K4の普遍的優位を主張しない。探索・確認をpoolしない。各metricのsample SD/SEは記述的で、未観測費用・familywise selectionを含むconfidence boundではない。

## 希少Taylor次数と主張限界

探索576 PR trajectoryで5040 eventを記録し、0次5024/2次15/4次1。独立確認192 PR trajectoryで3584 eventを記録し、0次3580/2次4/4次0。K4の4次は探索のq1/r1構成で一度観測したが、事前選定6構成の独立確認では未観測。q4/r4/K4の確認32本は全512 eventが0次、q8/r4/K4は全1024 eventが0次である。

これらのK4構成のP(order4)はq2/r4で1.282785e-5、q4/r4で8.121069e-7、q8/r4で5.092149e-8。発生確率が小さいことだけでは未観測full-wrapper費用の母平均寄与を保証しない。2次/4次の費用を0で補完せず、保存分布・観測counts・未観測massをartifactへ記録した。conditional rare samplingは追加していない。今回の差が見えることと、厳密な母平均上界を閉じたことを区別する。

数値allowanceは全92候補でEMPIRICALであり、ground-state・u・population winner certified flagsはfalse。全precision-status 368行中MISSING_COST/NUMERICALLY_UNRESOLVED0、INELIGIBLEはepsilon .05/.01/.005/.001で4/28/29/69。B0 prefix7の4qは全精度でINELIGIBLEなのでcompile対象外、592 groupの完了不足や失敗ではない。q8で点最良がdomain端にあるstrict精度は、q16以上の未探索最適性を主張しない。

## 状態準備費用の感度

epsilon .001ではPRの必要shots60,710,094、full PF36,390,063。PRは1wrapper平均RZ253,005.8125、full PF1,232,704で、shot増加を含めてもG(0)が低い。選定されたこの組を固定した共通状態準備RZ費用の交点は約1,212,917 RZ/shot。Pを含めると順位は変わり得るため、54–66%をend-to-end/state-preparation込みの削減率と扱わない。交点は固定点最良pairの感度であり、Pごとに全候補を再最適化したenvelopeではない。epsilon .01/.005の固定pair交点は約1.750e6/1.755e6、.05の現在pairではPRのshotsも少ないが、他PFへのP依存selectionは別。

## 完了・資源・lineage監査

Signal92/92、探索592/592、独立確認192/192。探索88適格構成の16det group＋72PR×8、確認6PR×32。successful full-wrapper1572件（paired primary1568＋ordinary control anchor4）、successful科学trajectory768件。全worker exit0/reaped・coordinator native exit0/wait-reap済み、最終run STOPなし・cleanup error0・own worker残存0。17:50:54 JSTにnative退出を観測、正確な退出timestampは未記録/null。

保存checkpoint metadata36,094 file SHAを照合し、task/seed/SOURCE/event digest/paired axes/全6metrics/compiler/native receiptと祖先reuseを確認。費用map3,576行を保存wrapperと固定shotから再算術して一致した。新signal/trajectory/compile calls0、NPZ科学array読込0、byte SHAのみ。実行SOURCE closure265件と入力5 file SHA一致。hash/cache再利用の実行と量子回路費用の改善を混同しない。実測speed倍率の比較試験は行っていない。

最終resume02の探索13時間2分57秒、確認1時間28分38秒、合計約14時間32分。全runの記録phase wall合計約16時間54分（引継ぎ・停止分を含む）。最初の起動から終了観測まで約20時間31分には修正/認可待ちの停止時間も含む。

最大worker RSS 7.077 GiB/VmPeak 7.743 GiB、最終run探索aggregate peak 11.547 GiB、確認aggregate peak 7.472 GiB。最大8worker・内部thread1、9/24GiBの二段階weighted admission・AS32/worker予約72/aggregate76/job96/5秒STOPを維持した。大きいgroupは予約待ちがあり常時8compileではない。

原run02の75 group・resume01の51 groupを新runへ受理し、原first USER_STOP_REQUESTと一時file FileNotFoundError停止を保存。known compile1572/trajectory768/occurrence3056/control360、held22/11/11/4を返却/resetせず保持。known+heldは1594/779/3067/364で、承認cap1722/843/4219/376内。累積carry付きのlisted lineageは3883 compile/1891 trajectory、今回承認枠4011/1955内。ただし旧H4 productionやunknown historical actualはこのsubtotalに含まず、プロジェクト全履歴の正確な消費数と呼ばない。

共有環境/既存venv/H4/他job変更なし、GPU query/use0、入力再生成0。旧profileとnative318 subsetの起動時照合を保持、旧host compilerの出力bit完全同一性は未確認のまま。計算は全core完了後mandatory STOPし、自動retry/次stageを起動していない。

## 保存資料と次の判断

[一つの結果入口](../../artifacts/resource_applicability/track_a_h8_resource_map_result_v1/2026-10-14/README.md)、[精度別CSV](../../artifacts/resource_applicability/track_a_h8_resource_map_result_v1/2026-10-14/precision_summary.csv)、[受領・完了監査](../../artifacts/resource_applicability/track_a_h8_resource_map_result_v1/2026-10-14/completion_audit.json)、[希少次数](../../artifacts/resource_applicability/track_a_h8_resource_map_result_v1/2026-10-14/rare_order_observation.json)。個々のwrapper RZ/CX/各depth/size・seed/source/compiler/event/fingerprintをcurated CSVへ保存、実runtime/checkpoint/cache/NPZはGitに含めない。

次の科学判断は、希少次数の費用保証をどこまで必要とするか、P込みcost・q8境界の評価、H4 rank12固定policyとH6/H8 tol-only policyを揃えたsize比較の範囲。今回追加sampling/transpile/q16/H4再計算を行わず、結果を固定して研究判断へ戻す。
