# ランダムRTE回路のcompiled-cost近似検証：統合結果

最終更新：2026-08-26 JST

## 1. この文書の役割

本書は、DF部分ランダム化研究で2026-08-23から2026-08-26に行ったランダムRTE回路の
compiled-cost検証を、一つの流れとして確認するための入口である。個々の検証条件、推定量、
標準誤差およびartifact fingerprintの正本は、リンク先の検証文書とmachine-readable JSONである。

本検証は、将来のRPE総コストへ入力する1回当たり期待compiled costの近似を対象とする。
量子shot、noise、実backend、状態準備および最終総コストは評価していない。

## 2. 検証した問いと現在の答え

ランダムevent列を $\omega=(e_1,\ldots,e_L)$ としたとき、求めたい量は

$$
\mathbb E[C_L]
=\sum_\omega p(\omega)C_{\mathrm{comp}}(\omega)
$$

である。大規模系または長いevent列では全trajectoryを一体compileできないため、短い局所回路から
この期待値を推定する必要がある。

現在の答えは次のとおりである。

1. eventを個別にcompileして足す単純加算は、境界をまたぐ相殺を失うため不採用とする。
2. 1--3 eventのconnected-cluster係数を条件ごとに較正し、解析的Taylor次数確率で再構成する。
3. 未使用full回路のholdoutで暫定5%点誤差と2%予測精度を確認する。
4. 3-eventまでで不通過となった条件だけ4-event係数を追加する。
5. 正確なDF Hamiltonian snapshotとcompiler条件をcalibrationの再現単位とする。

これはH4・H5の明示条件で得たlocal validation方針であり、H12、別compiler、coupling map付き回路、
または全RPE段へ無条件に一般化しない。

## 3. 記号の区別

Taylor cutoffとcluster次数を混同しないよう、次の表記を用いる。

- $K_{\mathrm{Tay}}$：有限Randomized Taylor Expansionの最大Taylor次数。
- $\kappa_1,\kappa_2,\kappa_3,\kappa_4$：1--4 event connected-cluster係数。

event列の局所予測は

$$
\widehat C_L(e_1,\ldots,e_L)
=\sum_i\kappa_1(e_i)
+\sum_i\kappa_2(e_i,e_{i+1})
+\sum_i\kappa_3(e_i,e_{i+1},e_{i+2})
+\cdots
$$

とする。$\kappa_2$は一境界の相殺・融合、$\kappa_3$は隣り合う二境界を独立に扱えない残差、
$\kappa_4$は3-eventまでで説明できない局所残差を表す。

## 4. 共通のcompiler条件と判定量

特に断らないH4結果は、H4 chain、原子間距離1.0 Å、STO-3G、DF rank 12、固定DF
Hamiltonian snapshot、Qiskit 1.3.0、basis gates `rz,sx,x,cx`、optimization level 1、
transpiler seed 17、coupling mapなしで得た。

cost metricはRZ countを主指標とし、RZ depth、CX count/depth、total depth、circuit sizeを
同時に検査した。運用推定器の暫定受理条件は次の二つである。

- 全metric最大絶対相対点誤差：5%以下。
- 主指標の予測側95%相対半幅：2%以下。

点ごとの正規近似95%上側値は保守的診断として記録するが、比率の厳密な信頼上界または
全metric同時被覆とは扱わない。absolute z-scoreが大きくても相対残差が小さい場合は、
「残差は非ゼロだが運用許容差内」と解釈し、残差ゼロとは主張しない。

## 5. 検証の経緯と結果

### 5.1 event単純加算の棄却

H4、$L_D=3$、$\delta=0.1$、$K_{\mathrm{Tay}}=0$、$r=2$の同一DF表現・100 trajectoryで、
event別compile和はRTE occurrence一体compileを全6 metricで48.30--57.58%過大評価した。
別DF表現の300 trajectory replicateでも52.99--61.15%の過大評価を確認した。

一方、決定論forward half、RTE occurrence全体、決定論reverse halfの3部分和は、同じ$r=2$
条件で一体partial-$S_2$を最大0.959%しか過大評価しなかった。したがって大きな差は主に
RTE event間のcompile境界を人工的に切ったことから生じる。

詳細：[ランダム回路compiled-cost加法モデル](random_circuit_cost_validation.md)

### 5.2 $K_{\mathrm{Tay}}=0$の局所境界補正

H4、$L_D=3$、short-step 0.025で、1--3 eventの係数を較正して未使用$L=4,6$へ予測した。
高統計の独立2 seedでは、pairまでのcount/size誤差が8.07--8.18%、最大zが約3で再現した。
tripleまで含めると全metric最大誤差は2.33--3.73%、最大zは0.97以下になった。

未使用$L=8$・2000 trajectoryでも、pair-onlyのcount/size誤差は最大8.851%残った一方、
triple予測は全metric最大1.744%、最大z 0.521だった。depthではpair-onlyの統計的不適合は
確認されていない。

同じfragmentとなる境界の解析確率は0.7310604だった。same-fragment補正だけでは最大3.65%、
最大z 2.59の差が残ったが、same/different双方の条件付きpair補正を解析確率で重み付けすると、
別seed pair holdoutを最大0.849%、最大z 0.587以内で再現した。same/different分類はpair補正と
別モデルではなく、平均pair係数を確率から再構成する内部層別である。

詳細：[境界補正pilot](rte_boundary_cost_validation.md)、
[fragment層別・高統計検証](rte_boundary_pair_validation.md)、
[階層model拡張](hierarchical_cost_validation.md)

### 5.3 $K_{\mathrm{Tay}}=2$のrare Taylor次数への対応

旧IID pilotでは1 eventのorder-2確率が $1.062925\times10^{-4}$で、8000 event位置中
order 2は1回しか現れなかった。このrunは$K_{\mathrm{Tay}}=2$内部の妥当性を示す根拠から除外した。

全Taylor次数patternを条件付きsampleし、解析確率で再結合した。独立に推定した大きな
$C_1,C_2,C_3$を差し引く500標本推定では最大点誤差9.119%、最大z 1.651となり、構造残差と
Monte Carlo分散を分離できなかった。一方、同一trajectory上でfull回路と1--3 event局所窓を
対応付けたpaired残差では、$L=4,6$のall-order-0とexactly-one-order-2を全metric最大1.373%、
RZの95%診断1.796%以内で再現した。最大z 7.535なので小さい4-event以上の残差は非ゼロだが、
代表条件の暫定5%内だった。

paired残差はcluster打切り構造の検証に用いる。実際の運用推定は、holdoutと独立した局所係数を
pilot/Neyman配分で較正し、解析Taylor次数確率で再構成する。

詳細：[階層model拡張](hierarchical_cost_validation.md)、
[connected-cluster運用検証](rte_connected_cluster_cost_validation.md)

### 5.4 H4の運用推定と条件移送

H4、$L_D=3$、short-step 0.025、$K_{\mathrm{Tay}}=2$の主runでは、all-order-0を各長さ1500、
exactly-one-order-2を各位置375 trajectoryとした未使用$L=4,6,8$ holdoutに対し、
全metric最大点誤差2.936%、主指標最大z 2.074、予測側95%相対半幅1.537%だった。

同じH4 snapshot・compilerで各条件を再較正した高統計transfer holdoutは次のとおりである。

| $L_D$ | short-step | 全metric最大点誤差 | 主指標最大z | 予測側95%半幅 | 判定 |
|---:|---:|---:|---:|---:|:---:|
| 0 | 0.025 | 2.043% | 1.430 | 1.569% | pass |
| 3 | 0.020 | 2.140% | 1.645 | 1.580% | pass |
| 3 | 0.025 | 3.777% | 2.784 | 1.579% | pass |
| 3 | 0.030 | 3.541% | 3.037 | 1.784% | pass |
| 6 | 0.025 | 6.119% | 5.213 | 1.625% | fail |

$L_D=0$の複数order-2を含む全patternについて、同一trajectoryのpaired残差で再検証すると
全metric最大点誤差1.679%、主指標95%診断3.201%となり、先行する独立係数推定の最大21.426%
超過はcluster構造の破綻を示さなかった。

$L_D=6$には独立4-event係数を追加した。固定$\kappa_1$--$\kappa_4$の未使用$L=8$評価は
全metric最大点誤差3.750%だったが、主指標95%上側診断は7.045%である。さらに同一trajectoryの
paired $\kappa_1$--$\kappa_4$構造検証を$L=8$で行うと、最大点誤差4.008%、主指標95%診断
4.902%となった。したがって4-event項はこの条件の5%点候補を回復するが、独立係数の統計的不確かさは
残るため、全条件へ一律に追加しない。

### 5.5 controlled反復の$q$依存

H4、$L_D=3$、$\delta=0.1$、$r=4$、$K_{\mathrm{Tay}}=0$、同一compilerで、controlled
partial-$S_2$の一体compile期待costを $\widehat D_q=a+bq$ で近似した。
$q=1,2$各300 trajectoryから未使用$q=4$を予測した最大誤差は0.307%、各150 trajectoryから
未使用$q=8$を予測した最大誤差は0.0529%だった。

対象はordinary controlled time-evolution subcircuitであり、ancilla Hadamard、軸変更、測定、
状態準備および量子shotは含めない。

### 5.6 H5への系サイズ方向の検証

H5 chain、原子間距離1.0 Å、STO-3G、10 qubit、project設定DF rank 9、$L_D=4$、
short-step 0.025、$K_{\mathrm{Tay}}=2$、同じQiskit/compiler条件で$L=4,6,8$を評価した。

同一trajectoryの構造検証では、$\kappa_1$--$\kappa_3$の全metric最大点誤差は1.665%、
$\kappa_1$--$\kappa_4$は0.551%だった。事前規則「5%を満たす最小cluster長」により3-eventまでを
選択した。all-order-0は各長さ100、exactly-one-order-2は各位置20 trajectoryである。

次に独立calibrationと独立full holdoutを実行した。calibrationはpilot 30、最大2000、RZ相対標準誤差
目標1%、adaptive最大2 roundで、実現最大相対標準誤差0.745%に到達した。holdoutはall-order-0を
各長さ500、exactly-one-order-2を各位置125 trajectoryとした。$\kappa_1$--$\kappa_3$の結果は、

- 全metric・主指標最大点誤差：3.776%。
- 主指標最大absolute z-score：2.009。
- 主指標最大予測側95%相対半幅：1.459%。
- 主指標最大点wise正規近似95%上側診断：7.461%。

であり、事前の5%点誤差と2%予測精度を通過した。95%上側診断は硬い受理条件ではなく、
rigorousな5%保証は主張しない。

## 6. 現在採用する運用規則

大きな系のscreening後にcompiled costを評価するときは、次の順序を採用する。

1. 同一Hamiltonian snapshot、$L_D$、short-step、$K_{\mathrm{Tay}}$、compiler/backend条件を固定する。
2. まず$\kappa_1$--$\kappa_3$を局所回路だけで較正する。
3. 保存係数と解析Taylor次数確率から長いevent列をtranspileなしで予測する。
4. 各新条件では代表的な未使用full回路holdoutを行う。
5. 5%点誤差または2%予測精度を外れた条件だけ$\kappa_4$または追加境界classを検証する。
6. 候補の推定cost差が約5%以下なら、追加samplingまたは直接full compileで順位を確認する。

検証済み範囲のモデル不確かさとして約5%を持たせる。これは厳密上界ではなく、候補順位がこの幅より
十分離れているかを判断するための運用値である。

## 7. 実装済みの軽量化

実装は次の三段階に分離した。

- calibration：局所1--3 event回路だけを構築・transpileする。
- prediction：保存済み係数と解析確率だけを使い、Qiskitを呼ばない。
- holdout：固定calibrationを変更せず、選んだfull回路だけをcompileする。

固定index sample chunk、十分統計量の増分合成、atomic checkpoint、精度未達層へのadaptive再配分、
exact numeric circuitをkeyとするSQLite metric cacheを実装した。cache、checkpoint、logは再開用の
runtime stateでありgitには含めない。結果JSON、status、plan、生成script、validator、専用test、
および正確なHamiltonian snapshotを証拠単位として保存する。

## 8. 未検証範囲と再検証のtrigger

回路コスト近似の広いpilot検証は一旦区切る。次の場合だけ追加検証する。

- 実際の最適化で必要なevent列長が8を超える。
- compiler version、optimization level、basis gates、coupling map、layout/routingまたはbackendを変える。
- 新しいHamiltonian snapshot、$L_D$、short-step、Taylor cutoffで3-event holdoutが5%を外れる。
- controlled反復を、現在の$q$範囲または回路scopeを越えて使う。
- 候補間の推定cost差が約5%以下で、model uncertaintyが順位を変え得る。
- 厳密に「95%で誤差5%未満」を主張する必要が生じる。

現在の次工程は、検証済みcost providerをRPE roundごとのshot数、誤差予算、失敗確率配分へ接続する
ことである。部分ランダム化の最終総コストまたは優位性はまだ評価していない。

## 9. 証拠map

| 段階 | 文書 | 主なmachine-readable evidence |
|---|---|---|
| event単純加算 | [random circuit cost](random_circuit_cost_validation.md) | `artifacts/random_circuit_cost_validation/` |
| pair/triple境界補正 | [boundary pilot](rte_boundary_cost_validation.md)、[fragment層別](rte_boundary_pair_validation.md) | `artifacts/rte_boundary_cost_validation/`、`artifacts/rte_boundary_pair_validation/` |
| $K_{\mathrm{Tay}}=2$・controlled $q$ | [hierarchical holdout](hierarchical_cost_validation.md) | `artifacts/hierarchical_cost_validation/`、`artifacts/rte_order_stratified_cost_validation/` |
| H4運用・transfer | [connected-cluster運用検証](rte_connected_cluster_cost_validation.md) | `artifacts/rte_connected_cluster_cost_validation/` |
| H4 follow-up | [2026-08-25研究ノート](research/研究ノート/2026-08-25.md) | `artifacts/rte_cost_data_batch/2026-08-25/`、`artifacts/rte_cost_followup_batch/2026-08-25/`、`artifacts/rte_cost_paired_k4_l8/2026-08-25/` |
| H5構造・独立holdout | [2026-08-26研究ノート](research/研究ノート/2026-08-26.md) | `artifacts/rte_cost_system_size_h5/2026-08-25/`、`artifacts/rte_cost_system_size_h5_independent/2026-08-26/` |

すべての新規数値はdirty local worktreeで生成したlocal evidenceである。このcommitに結果と生成コードを
固定しても、外部環境またはimmutable CIでの再実行が完了するまではexternally reproduced evidenceとは
呼ばない。
