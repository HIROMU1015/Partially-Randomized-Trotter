# C8-1 q64：保存8 whole-pathの完了・価格集計

2026-11-09 JST。**全8 pathのrawと全32 wrapper postcheck、最終paired X/Y・χが完了した。Referenceはこの有限8本でCX countを平均10.962%、total depthを平均54.588%、RZ countを平均0.987%削減し、logical幅は9→17となった。** ここでの削減率は方式別標本平均の比であり、全sampling lawや全RPEの優位性を保証しない。

## 固定条件とSOURCE

Linear H4／隣接1.00Å／STO-3G／Nα=Nβ=2／元H／number-CDF8 rank8／元frame順LD2／ordered partial S2/s1/r32/K0／h=δ=.1／q64／complete step32後global cut。元Hの厳密groundを仮定し、状態準備費を除外する。独立domainの固定8 whole-path、512 step occurrence、計16,384 eventsを使用した。X/Yは同じ8 pathで、16独立標本ではない。

- sampling・sample00 raw/4 P0 SOURCE：`6b1d613ded963b7851afa1c07ba8c646124dd44f`。初回q64の保存cohortを使用し、今回の限定完了でsamplingを追加していない。
- sample00数値修正postcheck SOURCE：`f1e89590c7daea12319352b85b959928ad166213`。同じ4 compiled QPY、同閾値、公式Aer doubleのsimulation fusion無効化。旧FAIL/STOPは保持。
- 限定完了SOURCE：`b1e7cc5b6270e6e62ab57b7c889f37484ce7d7b7`。sample00の最終paired/χ gate後、残7 raw・最大28 P0/postを実施した。
- packet SHA256：`1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`。CDF：`a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。law：`c347d7f2ec2ed40adb6f2490960dbebf693862cddcce96418c1027c48a68d6a5`。

実行結果は11月9日16:31 JSTに保存され、以後の本集計では再実行していない。保存された `paired_statistics.json` と `model_diagnostics.json` を再利用して表を整形した。固定実装・technical入力32 pin、8 event record hash、各path4価格recordのseed・ordered-event hash・cut対応を照合した。

## Correctness・call会計

| 項目 | 継承 | 新規限定完了 | 合計 |
|---|---:|---:|---:|
| sampling | 保存8 path | 0 | 保存cohortのみ |
| raw correctness | sample00 1 | 7 | 8/8 PASS |
| full-wrapper P0 | 4 | 28 | 32/32完了 |
| compiled postcheck | 4 | 28 | 32/32 PASS |
| 最終paired X/Y・χ | 0 | 8 | 8/8完了 |

AB=V、actual A†、半pathの真空ray、whole-path χ、scalar/identity、signed係数、Strong両branch・complex coherence・X/Y誘導observableとreference leakageを既存CAR/state-actionで検査した。全256 system入力とReferenceの全出力振幅を保ち、17-qubit全Operatorは構築しない。

| 保存checkからの最大値 | 値 |
|---|---:|
| raw_max_component | 4.30766534e-13 |
| post_max_component | 6.75360705e-11 |
| compiled_vs_CAR_or_raw_observable_max_Frobenius | 5.9955506e-11 |
| final_paired_XY_max_Frobenius | 5.46501353e-11 |
| vacuum_chi_max_component | 2.86402123e-13 |
| reference_leakage_max_component | 8.32414398e-13 |
| reference_leakage_max_L2 | 9.39911568e-13 |

固定component abs1e−10、observable Frobenius/signal2e−10、rtol0。phase fitting・再正規化・入力列削減・閾値緩和なし。表のobservable最大値はCAR/raw observableラベルに限定し、component基準のisometry Frobeniusや負のχ対照を混ぜていない。数値PASSは有限binary64の実装証拠で、rigorous enclosureではない。 継承sample00の4 postcheckにはfusionフラグの項目がないため固定f1e89590 SOURCEのfalseを照合し、新28 recordは各自のfalseを直接確認した。精度・phase fit・再正規化の不変更は全32 recordで直接確認した。

## Full-wrapper資源

P0：Qiskit1.3.0／opt1／seed17／rz,sx,x,cx／approximation_degree1／qubits_initially_zero=False／topology-free／既存宣言global phase保持。Strongはuncontrolled Gaussian・directional・head relative・tail basis共有・63 step境界融合・scalar集約を維持。Referenceはwhole-path一gadget、B前32/A後32、actual A.inverse、χ補正、二CSWAP cascade、内部62融合とglobal cutの1境界融合損失を含む。両方式とも強い同条件のfull Hadamard interrogationを価格化した。

以下はX/Yそれぞれ同じ値であることを保存recordで確認した。SDはddof1。Δ=Reference−Strong、減少率は100×(Strong平均−Reference平均)/Strong平均。qubitsの増加は費用の削減と相殺せず別指標にする。

| 資源 | Strong平均 ± SD | Reference平均 ± SD | paired Δ平均 ± SD | 観測Δ min..max | 平均比の減少率 |
|---|---:|---:|---:|---:|---:|
| RZ count | 336,797.375 ± 4,702.41 | 333,473.75 ± 4,697.938 | -3,323.625 ± 10.405 | -3,342 .. -3,310 | 0.987% |
| CX count | 128,880 ± 1,493.763 | 114,752 ± 1,493.763 | -14,128 ± 0 | -14,128 .. -14,128 | 10.962% |
| RZ depth | 81,961.125 ± 1,341.152 | 39,144.5 ± 1,032.636 | -42,816.625 ± 531.922 | -44,013 .. -42,328 | 52.240% |
| CX depth | 63,156.75 ± 662.213 | 23,925 ± 509.747 | -39,231.75 ± 273.215 | -39,832 .. -38,958 | 62.118% |
| Total depth | 190,931.875 ± 2,866.374 | 86,706.25 ± 2,214.168 | -104,225.625 ± 1,156.361 | -106,838 .. -103,139 | 54.588% |
| Circuit size | 688,284.875 ± 9,624.813 | 671,025.25 ± 9,620.182 | -17,259.625 ± 10.405 | -17,278 .. -17,246 | 2.508% |
| Peak logical qubits | 9 ± 0 | 17 ± 0 | 8 ± 0 | 8 .. 8 | 幅+8 |

RZ/CX count、両depth、total depth、sizeは8 pathすべてでReferenceが減少した。RZ countをT countへ換算せず、topology-free depthを実機時間へ換算しない。qやpath/cut・融合条件が異なる過去結果に対し、差をqだけの因果効果としない。

絶対価格の方式/資源/axis共分散28×28とpaired6資源/axis共分散12×12を元JSONに保持した。n8によるrank上限7、X/Y依存、幅とCX差の観測SD0を区別する。SD0・観測min/maxを全lawの分散0・support・tail boundと扱わない。

## CX整数候補と凍結モデル

全8 path・X/YともΔCX=−14,128で、候補208−224q（q64）の整数残差ΔCX+14,128は全て厳密整数0。A/Sモデルの浮動小数点予測差に現れる約1.82e−12はこの整数残差と区別する。全path/全q公式の認定はしていない。

主A=C7-4 all-q affine、F=元C6a frozen affine、S=C7-4 design[1,q,sqrt(q)]を変更せず評価した。主Aの指定は維持する。以下はX/Y共通のpaired差予測と保存平均の残差。

| 指標 | モデル | paired予測 | 観測Δ平均 | 観測−予測 |
|---|---|---:|---:|---:|
| CX count | A | -14128 | -14128 | 1.8189894e-12 |
| CX count | F | -14128 | -14128 | 0 |
| CX count | S | -14128 | -14128 | 1.8189894e-12 |
| Total depth | A | -103542.753 | -104225.625 | -682.872228 |
| Total depth | F | -97814.4375 | -104225.625 | -6411.1875 |
| Total depth | S | -104149.536 | -104225.625 | -76.0890234 |

CXの絶対平均に対する残差はStrong/Reference共通でA +455.122、F +5,024.625、S +8.171。Total depth絶対残差はAでStrong +870.472/Reference +187.600、F +10,656.313/+4,245.125、S −91.883/−167.972。このholdoutのCX/depth点残差はSで小さいが、モデル選択の確認的結論や母平均保証ではない。paired CX一致だけでは絶対価格の正しさを示さない。

今回の固定PLANは5%幅・α family・確認的PASSを採用していない。保存校正varianceとholdout標本mean SDを別欄に保持し、合成した信頼区間・総bias boundを新たに作っていない。全6資源×3モデル×2方式+paired×X/Yの108保存診断と共分散は元JSONを参照する。係数再fit・bootstrap追加・価格幅の事後変更は0。

## 原本・公開・未解決事項

run04のSOURCE・result・STOP・既存scalar JSON202件と固定SOURCE pin32項目は本集計の前後でbytes不変。sample00の別記 `paths/q64_sample00/*_price_metadata.json` 4件も元bytesで公開し、継承check hash一致・保存価格metrics一致・postcheck再実行falseを確認した。event/QPY/NPZ/conditions/fusion/内部進捗はlocalのみ、公開はscalar/check/統計/SOURCE/hash索引と本報告に限定する。元packet/CDF/law等はrun終了時の既存preflight照合とsource_input_preserved=trueを継承し、本集計では固定実装・technical入力32 pinと8 event recordをhash再確認した。大きいprivate QPY/NPZのhash索引は実行時の保存値を再利用し、不要な全件rehashや再アーカイブは行わない。

未解決は、母平均・全support/tail・実装law距離、長q外挿bias、q≥128での同精度検証、routing/T/noise、actual RPE成功保証。q64を用いた再fit、N再算定、全13-round総費用更新、詳細科学レビューは行っていない。

資料：同directoryの固定 [COMPLETION_PLAN](COMPLETION_PLAN.md)、[旧数値修正結果](NUMERICAL_REPAIR_RESULT.md)、[更新長q戦略](LONG_Q_STRATEGY.md)。機械可読資産は `artifacts/track_d_c8_1_q64/2026-11-08/run04_limited_completion/` のprices/statistics/model_diagnostics/cx_candidate_residuals/payload_index/aggregation_20261109を参照する。

次はユーザー／GPTがCX構造bridge・depthの外挿bias・高寄与長qの独立検証と数値保証の範囲を判断する。Codexは公開SHA/tree/bytesを照合して **mandatory STOP**。追加q・新sampling・科学レビューへ自動進行しない。
