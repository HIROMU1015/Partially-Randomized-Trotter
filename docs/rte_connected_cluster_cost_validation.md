# RTE connected-cluster compiled-cost validation

## 目的と範囲

ランダムRTE event列をすべてコンパイルせずに、短い局所回路のcompiled costから
長いevent列の期待costを推定できるかを検証する。これは将来の総コスト評価へ入力する
近似modelの局所検証であり、full RPE、量子shot、noise、実backend、または最終総コストを
評価したものではない。

主結果の条件はH4 chain、原子間距離1.0 Å、STO-3G、DF rank 12、`L_D=3`、
`delta=0.1`、RTE short-step数`r=4`、finite Taylor cutoff `K=2`である。
compilerはQiskit 1.3.0、basis gates `rz,sx,x,cx`、optimization level 1、
transpiler seed 17、coupling mapなしの1条件に限定する。

## 推定model

event列`e_1,...,e_L`のcost `G`を、最大3 eventのconnected係数で

\[
\widehat G(e_1,\ldots,e_L)
=\sum_i\kappa_1(e_i)
+\sum_i\kappa_2(e_i,e_{i+1})
+\sum_i\kappa_3(e_i,e_{i+1},e_{i+2})
\]

と近似する。局所係数は同一trajectoryの短い窓を対応付けて

\[
\kappa_1(e_1)=G(e_1),\qquad
\kappa_2(e_1,e_2)=G(e_1,e_2)-G(e_1)-G(e_2),
\]

\[
\kappa_3(e_1,e_2,e_3)
=G(e_1,e_2,e_3)-G(e_1,e_2)-G(e_2,e_3)+G(e_2)
\]

から求める。Taylor次数0と2の全patternを条件付きで直接sampleし、実際の低いorder-2確率を
待たず、最後に解析的order確率で再構成する。pair係数は境界がsame basisかdifferent basisかで
層別し、その解析確率で再結合する。order-0単eventは全componentを厳密列挙する。

pilotからRZ count予測の相対standard error 1%を目標にNeyman配分し、最低30、最大1500標本、
safety factor 1.5とした。productionとholdoutはseedを分離し、holdoutを係数調整には使わない。

## 独立holdout

同一の浮動小数点DF Hamiltonian snapshot上で、未使用のfull回路を次の条件で評価した。

- event列長`L=4,6,8`
- 全eventがTaylor order 0: 各長さ1500標本
- order 2がちょうど1 event: 各位置375標本
- cost metric: RZ count/depth、CX count/depth、total depth、circuit size

主metricはRZ countとし、事前の点推定許容差を5%、予測側95%相対半幅を2%とした。
点ごとの正規近似95%上側5%は保守的診断であり、rigorous boundではない。

## 結果

主artifactは
[`h4_...hold1500_rare375_v1.json`](../artifacts/rte_connected_cluster_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.json)
である。同じrunのDF配列は隣接する`.hamiltonian.npz`へ保存した。

- RZ countの最大絶対相対点誤差: 2.936%
- 全6 metricの最大絶対相対点誤差: 2.936%
- RZ countの最大absolute z-score: 2.074
- RZ countの最大予測側95%相対半幅: 1.537%
- RZ countの最大点wise正規近似95%上側診断: 5.724%
- production sample cap: `k1:order-2`のみ、要求1535に対して1500

最大点誤差は`L=8`、全order-0のRZ countで、予測979.753、full回路平均951.811だった。
全order-0 RZ countの符号付き差は`L=4`で+2.59%、`L=6`で-0.35%、`L=8`で+2.94%であり、
長さとともに単調増加していない。order-2がちょうど1回のRZ count点誤差は全長さで
1.02%以下だった。したがって4-event係数を直ちに追加する兆候は弱い。

点推定5%と予測精度2%は通過した。一方、保守的95%診断は5.724%で未達であり、
sample capにも僅かに到達した。この結果は「代表1条件で、実用点誤差5%内の暫定候補」とし、
rigorousな5%保証またはcompiler条件を越えた一般化とは扱わない。

## Hamiltonian再構築のイレギュラー

先行1000/250標本artifactへ、別processで同じ分子条件から再構築したHamiltonianのholdoutを
追加しようとしたところ、元holdoutと追加holdoutの全metric最大差は11.68%、最大z-scoreは
5.215だった。HF energyやRTE order確率の差は丸め誤差程度でも、DF表現の符号・縮退部分の
基底選択などがcompiled costへ影響し得る。したがって、この追加標本は主結果へ結合していない。

以後のcompiled-cost検証では、分子条件だけでなく正確なDF Hamiltonian配列をsnapshot保存し、
calibration、holdout、追加runで同一snapshotを使う。主1500/375 runはこの方針で内部整合する。

## 再開と再現性

generatorはHamiltonian snapshotと、pilot K1--K3、production K1--K3、holdout L4/L6/L8の
9個のfingerprinted checkpointを保存する。同じcommandを再実行すると、task fingerprintが
一致する完了checkpointを読み、未完了taskだけを実行する。

```bash
.venv311/bin/python scripts/run_rte_connected_cluster_cost_validation.py \
  --holdout-zero-samples 1500 \
  --holdout-single-order2-samples 375 \
  --seed 20260827
```

artifact、snapshot、checkpointはdirty local worktree evidenceでありimmutable CI evidenceではない。

## 2026-08-25追加データ採取

未解決範囲を切り分けるため、同一の主snapshotを再利用する三つのjobを追加した。

- $L_D=0$で、order-2が2回以上現れる$K=2$条件を含む全order patternのpaired residualを測る。
- $L_D=6$で、4-event窓のfull/triple/pair/singleを同じtrajectoryに作用させ、独立K4係数を直接測る。
- controlled partial-$S_2$の$q=1,2$較正を未使用$q=8$でholdoutする。

`scripts/run_rte_cost_data_batch.py`は前二者を並列化し、その後に$q=8$を実行する。
進行状況、log、checkpoint、候補成果物は`artifacts/rte_cost_data_batch/2026-08-25/`へ分離する。
この節は実行計画と実装状態を記録するもので、完了後の数値結果やK4採用を主張しない。

追加batch完了後、複数order-2を独立係数差分だけで評価すると不確かさが大きいこと、
$L_D=6$では独立paired K4により最大点誤差が1.281%へ下がる一方、30標本ではRZの
95%上側診断が5.714%であること、controlled $q=8$ holdoutは全metric最大0.0529%であることを
確認した。そこでfollow-upでは、複数order-2を同一trajectoryのpaired残差へ置き換え、K4を
500標本へ増やし、係数を固定した未使用L8 holdoutを追加する。結果生成前は実装・計画として扱う。

## 総コスト探索向けの軽量運用実装

主1500/375 holdoutの科学的結果は上記v1 artifactに固定したまま、総コスト探索でfull回路を
毎候補compileしないためのv2実装を追加した。処理を次の三段階へ分離している。

1. `calibrate_connected_cluster_cost_model`：pilotと1--3 eventの局所係数較正だけを実行する。
2. `predict_connected_cluster_cost`：保存済み係数と解析的Taylor次数確率から任意のevent列長を
   予測する。この関数はQiskit回路を構築・transpileしない。
3. `validate_connected_cluster_calibration_holdout`：固定したcalibrationを変更せず、必要な条件だけ
   未使用full回路で検証する。

calibrationとtransfer validationは別fingerprintを持ち、正確なDF snapshot hash、`L_D`、
short-step時間、Taylor cutoff、compiler、coupling条件が一致しない結果を再利用しない。
generatorは`--mode calibration`を持ち、transfer専用scriptとして
`scripts/run_rte_connected_cluster_transfer_validation.py`を追加した。

性能面では次を実装した。

- 14個のTaylor patternをpattern単位に分け、さらに標本列を固定indexのsample chunkへ分割する。
  generatorの既定chunk幅は128で、各chunkを独立したatomic checkpointへ保存する。
- holdoutも長さ単位ではなく、全order-0とorder-2位置ごとのpattern・sample chunkへ分割する。
- pairのsame/different基底をrejectionで待たず、解析確率から直接条件付きsampleする。
- Neyman配分ではcacheやworker schedulingに依存するwall timeを使わず、局所窓数とevent
  application数から作る決定論的relative workを用いる。
- 完全な数値回路fingerprint、compiler hash、backend fingerprintをkeyとするSQLite metric cacheを
  追加する。数値角度をkeyから除外していない。
- 各chunkのseedはbase seed、pattern、chunk indexから決定論的に導出する。標本数を増やした場合、
  完了済みのfull chunkはそのまま再利用し、旧末尾のpartial chunkだけを置き換えて新規chunkを追加する。
  各chunkのcount、mean、標本分散を十分統計量として合成するため、既存回路の再生成も不要になる。
- same/different基底で層別したpairは、class別のchunk統計量を先に合成してから解析確率で再重み付けする。
  これによりchunk境界に依存せず、従来の層別推定量と同じ量を推定する。
- 旧v2のpattern全体checkpointは、条件と要求標本数が完全一致する場合に読み取り専用で再利用する。
  一致しない旧summaryへ新標本を追記することはしない。
- pilot配分後のproductionについて、実現した最大相対標準誤差を再評価し、未達なら分散から
  Neyman配分を更新して不足層だけを追加するadaptive roundを実装した。generatorの既定は最大2 roundで、
  到達、標本cap、配分不変、round上限の停止理由をartifactへ保存する。
- task fingerprintには入力条件だけでなくtask実装versionも含める。実装versionが変わった場合、
  旧checkpointは削除せずに残し、新しいsuffix付きcheckpointへ再計算する。
- chunkの性能記録は最大chunk時間、worker時間合計、pilot/production/holdout stageの実経過時間を
  分離して保存する。
- v2 calibration/transfer artifactは全体fingerprintに加え、condition、Hamiltonian、preparation、
  partition、compiler、distribution、seedの内部対応をvalidatorで検査する。

H4の固定snapshotに対する実装smokeでは、pilot 2、production cap 4のcold calibrationが約39.8秒、
別checkpointから同じ回路を読むwarm runが約37.3秒だった。warm runのtranspile missは0だったが、
order-0全componentのQiskit回路構築と厳密fingerprint計算が残るため、永続cache単独のwall-time短縮は
小さい。通常の同一checkpoint再開では完了task自体を読み飛ばす。

回転角をまたぐ構造再利用を判断するため、同じevent identityを複数short-step時間でcompileする
`validate_rte_cost_angle_invariance`も実装した。H4、`L_D=3`、short-step時間
0.020、0.025、0.030、各Taylor pattern 2標本のsmokeでは全6 metricが一致したが、これは
低標本の実装確認にすぎない。別`L_D`、十分なsame/different境界coverage、compiler/coupling条件を
検証するまで、構造cache再利用は明示的に無効のままとする。

固定chunk範囲、統計量合成、旧checkpoint互換、標本数増加時のfull chunk再利用、worker数による
科学的出力の不変性、chunk seed改変の拒否を回帰testへ追加した。変更後のlocal全test suiteは
`468 passed, 4 warnings`だった。上記smokeは`/tmp`上で行い、
新しい科学的artifactまたは主結果としてmanifestへ登録していない。

## 固定snapshot上の移送holdout

同じH4 chain、距離1.0 Å、STO-3G、DF rank 12、同じHamiltonian snapshot、finite Taylor
cutoff $K=2$、同じQiskit/compiler条件を保ち、$L_D$またはshort-step時間を変更して
K1--K3係数を各条件で較正した後、未使用$L=4,6$ full回路へ移送した。高統計runは全order-0を
各長さ1500標本、order-2が1回の条件を各位置500標本とした。

| $L_D$ | short-step | 全metric最大点誤差 | RZ最大z | 予測側95%半幅 | 5%点基準 |
|---:|---:|---:|---:|---:|:---:|
| 0 | 0.025 | 2.043% | 1.430 | 1.569% | pass |
| 3 | 0.020 | 2.140% | 1.645 | 1.580% | pass |
| 3 | 0.025 | 3.777% | 2.784 | 1.579% | pass |
| 3 | 0.030 | 3.541% | 3.037 | 1.784% | pass |
| 6 | 0.025 | 6.119% | 5.213 | 1.625% | fail |

$L_D=3$、short-step 0.030では、500/150標本の予備runが全metric最大7.138%、RZ最大5.754%、
予測側95%半幅2.265%で点・精度基準とも不通過だった。1500/500標本へ増やすと全metric
3.541%、RZ 3.488%、予測側95%半幅1.784%となり、予備runの超過は高統計で維持されなかった。
この再開runではcalibrationの相対標準誤差目標も1.0%から0.8%へ締めたため、改善をholdout標本数
だけの効果には帰属しない。採用した運用条件全体として点・精度基準を通過したと解釈する。
ただし最大zは3.037、点ごとの正規近似95%上側診断はRZで5.738%であり、残差ゼロやrigorousな
5%保証は主張しない。

$L_D=6$の高統計不通過はz=5.213を伴うため、単なる標本不足とは扱わない。$L=4$残差から
4-event係数をfitして$L=6$へ適用する診断では最大点誤差が2.990%へ下がったが、共有calibrationの
共分散を評価していないため、K4採用の最終受理結果ではなくdiagnosticに留める。

short-step 0.030の再開runは、旧pattern checkpointとv3 sample chunkを併用し、8000 full回路のうち
1250件をpersistent cacheから再利用した。3 workerの実経過時間は393.2秒だった。これは増分再開の
局所性能記録であり、大規模系または別compilerで同じ速度を保証しない。

## 2026-08-25 H4 follow-up

同じH4 fixed snapshot、Qiskit/compiler条件、finite Taylor cutoff $K=2$を使い、先行transferで
残った二つの問題を再検証した。

### 複数order-2

$L_D=0$、$L=4,6$の全Taylor patternについて、同一trajectoryのfull回路と1--3 event局所窓を
対応付けた。all-order-0とsingle-order-2は各pattern 10、multi-order-2は各pattern 20標本である。
全metric最大点誤差は1.679%、主RZ最大absolute z-scoreは6.703、主RZの点wise正規近似95%上側
診断は3.201%だった。先行する独立K1--K3係数の最大21.426%超過は、複数order-2に固有の
cluster構造破綻ではなく、大きな独立平均の差引きに伴う不確かさを含んでいた。

### $L_D=6$のK4

$L_D=6$、short-step 0.025の4-event係数を各Taylor pattern 500標本へ増やし、固定K1--K4を
未使用$L=8$のall-order-0 1500、single-order-2各位置500標本へ適用した。全metric最大点誤差は
3.750%、主RZ最大zは2.290、主RZの95%上側診断は7.045%だった。

さらに、all-order-0 100、single-order-2各位置20標本で、同一trajectoryのK1--K4局所予測と
full $L=8$回路を直接比較した。全metric最大点誤差4.008%、主RZ最大z 10.630、主RZ 95%上側
診断4.902%だった。したがって4-event項はこの不通過条件の5%点候補を回復するが、独立係数の
統計的不確かさを含むrigorousな5%保証ではない。K4は全条件へ一律に追加せず、K1--K3 holdoutが
不通過となる条件だけで較正する。

### controlled $q=8$

$L_D=3$、$\delta=0.1$、$r=4$、$K=0$で、$q=1,2$各150 trajectoryから得たaffine modelを
未使用$q=8$の150 trajectoryへ適用した。全metric最大点誤差は0.0529%、RZは0.0280%、主RZ
95%上側診断は1.091%だった。対象はordinary controlled time-evolution subcircuitであり、
Hadamard、軸変更、測定、状態準備およびquantum shotを含まない。

## H5の系サイズ方向holdout

H5 chain、距離1.0 Å、STO-3G、10 qubit、project設定DF rank 9、$L_D=4$、short-step 0.025、
$K=2$、同じQiskit/compiler条件で$L=4,6,8$を評価した。exact DF snapshotをpaired構造検証と
独立運用holdoutで共有した。

同一trajectoryのpaired residualはall-order-0各長さ100、single-order-2各位置20標本とした。
K1--K3の全metric最大点誤差は1.665%、主RZ 95%診断2.038%、K1--K4は0.551%、0.737%だった。
事前規則「5%を満たす最小cluster長」によりK1--K3を選んだ。

独立運用runではpilot 30、production cap 2000、RZ相対standard error目標1%、adaptive最大2 roundで
較正し、実現最大RZ相対standard errorは0.745%だった。holdoutはall-order-0各長さ500、
single-order-2各位置125標本とした。全metric最大点誤差3.776%、主RZ最大z 2.009、予測側95%
相対半幅1.459%、点wise正規近似95%上側診断7.461%だった。事前の5%点誤差と2%予測精度を
通過したので、このH5 snapshot・compiler条件では独立K1--K3運用推定器を受理する。95%上側診断は
硬い判定ではなく、rigorousな5%保証とは扱わない。

## 次の扱い

このcompilerに対するH4/H5の広いpilot検証は、上記留保付きで一旦区切る。K1--K3を全条件で
無条件に採用せず、exact snapshot・compiler条件ごとに代表holdoutを確認し、5%点誤差または2%
予測精度を超える条件だけK4または追加境界classを検証する。実際に必要な列長が8を超える、
coupling map/compiler/backendを変更する、controlled $q$または回路scopeを拡張する場合に
追加holdoutを行う。次工程は、このcost providerをRPE shot数と誤差・失敗確率配分へ接続することであり、
最終総コストは未評価である。
