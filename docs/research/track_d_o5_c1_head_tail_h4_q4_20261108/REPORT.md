# Track D O5 C1：保存実H4・q4のhead-tail融合結果

2026-11-08 JST。**指定一pathで、位相・全入力信号を保持した追加P0資源削減を両方式に観測した。** X/YともStrongはCX −30、Total depth −178、ReferenceはCX −30、Total depth −81。Referenceの対Strong CX差は不変で、depth差は97縮まった。Reference固有の相対利益が増えたとはいえない。

## 固定条件・SOURCE・保存原本

Linear H4／隣接1.00 Å／STO-3G／Nα=Nβ=2／元H／Number-CDF8（rank8固定）／元frame順L_D2／ordered partial S2/s1/r32/K0／h=δ=.1／q4・complete step2後cut。元H厳密ground利用を仮定し準備費を除外。対象はC6a calibration cohortの保存 `q4_sample00` 一件だけ。

科学実行SOURCE：`0df0134b5f259bbb2603719227e21831d876af9f`。人工基点：`14276fe0e2dadef54ef105debe6d6d99b43b6ecb`。保存C6a SOURCE：`6b6667026ec8b670f610e41141d195883a9eb513`。以前の原本回収・再利用確認：`2f316fb229149117177367720a6eaa5aa8a94b76`。保存結果だけの要約完了SOURCE：`e8a2b14e99c8b8f89f5a84788b982b8962f85ac6`。

- 元H packet SHA256：`1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`
- CDF endpoint SHA256：`a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`
- law SHA256：`c347d7f2ec2ed40adb6f2490960dbebf693862cddcce96418c1027c48a68d6a5`
- archive SHA256：`a631cd394073f7770d2723eb0cfa93fc6a8f703534b9cdce88b7c126381a007b`
- events SHA256：`919f2e784920581380cb4090c59fff34741d914e9d340d249bca554eba8cadb1`

全20 archive member、旧raw全256入力PASS、旧4 compiled QPY／postcheck／価格、request/event identity、scalar／identity／χ、signed rotation、P0、qubit/ancilla/readout規約を資格付けた。OFF core・A/B・actual inverseを宣言phase込みのcanonical IRで再構成照合し、保存S0/R0 X/Yを適格再利用した。旧4本のcompile／postcheck再実行は0。原本を再生成していない。

## 8-mode移送とcorrectness

新規opt-in adapterは人工版の2-mode条件を実8-mode／Strong9-wire／physical8-wireの明示guardに拡張する。既存StrongBuilder.transition・C3.prepare_steps/physical_segment・full-frame Provider・reference_gadget・MPFR160 compiler／既存checkerを呼び、元C6a defaultは変更しない。Gaussian合成器・Hamiltonian builder・sampler・state simulator・gadgetの新実装はない。

対象はhead出口→tail最初のfull-frame入口だけ。actual native span、rotation、wire順0..7、analytic gaugeとcontrol modeを照合しG_T†G_Hへ置換した。両方式各4接口を融合、拒否0。step0/1/2はblock2→3、step3は2→6。各局所spanは64→32 native命令だが、これを足してcompiled価格を算定していない。Strongの3 step境界融合とReferenceの2内部融合／global cut損失1・二層CSWAPを維持し、融合後Aからactual A.inverse()を作った。

| 検証 | 結果 |
|---|---|
| 実行前人工回帰 | 新8-mode6＋既存13＝19 PASS |
| 実8-mode局所bridge | 2種×全Fock位相／独立exterior lift＝4 PASS、最大成分差5.573e−15 |
| 全256入力raw | 既存22項目＋paired X/Y2 PASS、成分最大3.154e−14 |
| 融合前後・Strong追加照合 | A/B/actual inverse・Strong full512 operator・W0†W1=Vの5項目PASS |
| 新P0 compiled postcheck | 全4 wrapper・18比較PASS、最大成分差4.360e−12、observable Frobenius最大2.263e−12 |
| 保存X/Yの最終集約 | paired・full complex coherence・analytic χの6比較PASS、Frobenius最大2.378e−12 |
| Reference leakage | raw成分0、compiled最大成分9.437e−14 |

固定component abs1e−10、observable Frobenius/signal2e−10、rtol0、global phase fittingなし。AB=V、各half vacuum、actual inverse、scalar/identity・signed RTE、Strong両branch／coherence、X/Y誘導observable、bit規約を検査した。Referenceは全256 system入力と各131072出力成分を既存state-actionで保持し、17-qubit Operatorは構築していない。headerのglobal_phase値だけで同値性を判断せず、位相込み作用を比較した。

## P0 full-wrapper全7資源

Qiskit1.3.0／optimization_level1／seed17／rz,sx,x,cx／approximation_degree1／qubits_initially_zero=False／topology-free／既存MPFR160宣言phase保持。状態準備なしHadamard 1 shot全体の比較で、測定・Reference registerを含む。

| 構成 | axis | RZ count | CX count | RZ depth | CX depth | Total depth | Size | Qubits |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S0 | X | 21020 | 8024 | 5013 | 3874 | 11671 | 42839 | 9 |
| S0 | Y | 21020 | 8024 | 5013 | 3874 | 11671 | 42839 | 9 |
| S1 | X | 20997 | 7994 | 4932 | 3836 | 11493 | 42710 | 9 |
| S1 | Y | 20997 | 7994 | 4932 | 3836 | 11493 | 42710 | 9 |
| R0 | X | 21239 | 7336 | 2571 | 1590 | 5686 | 42562 | 17 |
| R0 | Y | 21239 | 7336 | 2571 | 1590 | 5686 | 42562 | 17 |
| R1 | X | 21216 | 7306 | 2534 | 1572 | 5605 | 42433 | 17 |
| R1 | Y | 21216 | 7306 | 2534 | 1572 | 5605 | 42433 | 17 |

X/Yは同じpathの二軸で、独立2標本ではない。

| 資源 | S1−S0 | R1−R0 | R0−S0 | R1−S1 | 差の差 |
|---|---:|---:|---:|---:|---:|
| rz_count | -23 | -23 | 219 | 219 | 0 |
| cx_count | -30 | -30 | -688 | -688 | 0 |
| rz_depth | -81 | -37 | -2442 | -2398 | 44 |
| cx_depth | -38 | -18 | -2284 | -2264 | 20 |
| total_depth | -178 | -81 | -5985 | -5888 | 97 |
| circuit_size | -129 | -129 | -277 | -277 | 0 |
| qubit_count | 0 | 0 | 8 | 8 | 0 |

差は両軸で同じ。**A：Strong自身は全count/size/depthで改善、幅9不変。B：Reference自身も全count/size/depthで改善、幅17不変。C：Referenceの相対CX/count/size差は不変で、RZ depth差44、CX depth差20、Total depth差97だけ縮小した。** StrongのTotal depth削減は約1.525%、Referenceは約1.425%。Referenceは融合後もCXが688、Total depthが5888小さいが、これは既存の利点を含み、今回の新しい効果として二重計上しない。RZはReferenceが219多く、logical幅も8多い。

局所bridge命令数を合算せず、実際の全wrapper P0と既存DAG/counterを用いた結果である。P0後にも追加利益が残る一pathの観測で、compilerの各passや各接口への因果的割当・最適性・他pathへの一般性は未取得。RZ→Tやtopology-free depth→実機時間へ換算しない。

## 技術的停止と保存結果の完了

run01は全4 postcheck PASS後、最後のχ要約でshape(1,)を行列用Frobenius checkerへ渡したため停止した。失敗は `Invalid norm order 'fro' for vectors` であり、correctness閾値超過ではない。旧SOURCE・87ファイル・private原本・FAILURE・STOPをそのまま保全し、[summary専用PLAN](SUMMARY_COMPLETION_PLAN.md)に従って保存済みobservableの同じχ値を1×1行列へ包み、未完了集約だけを別SOURCE／namespaceで完了した。科学実行・compile・wrapper postcheckをやり直していない。

科学callの合算：保存path1、event decode1、新sampling0、新S1/R1 core各1、新full wrapper4、raw bundle1、新P0 4、新compiled postcheck4、旧baseline4再利用。summary完了の新build/raw/P0/postcheckは全0。ground／DF fit／価格モデルfit／N／全13-round／別path／科学retryは全0。

## 次の判断に必要な材料と限界

1. 実H4への数学的・数値的移送：この保存pathのactual Gaussian・位相・全入力作用は成立した。任意8-mode問題の厳密証明ではない。
2. P0後追加利益：両方式で観測した。人工2-modeのdepth増分が等しかったことをH4へ仮定せず、異なるdepth削減を実測した。
3. 利益の帰属：両方式共通のC1最適化。depthではStrongの絶対削減が大きい。
4. Reference相対利益：新たな増加なし。CX差は不変、depth gapは縮小した。
5. C1継続価値：一pathでの実装可能性と小さい追加利益は得た。次に進むなら代表性・他pathでの再現範囲が論点だが、今回追加実行を許可・開始していない。
6. O1/O2/O3共同最適化：Reference固有の追加優位を本結果だけから期待する根拠はない。C1を共通局所passとして扱い、PR全体の改善余地と比較する判断材料になる。重点移動の正式採択・新実装・全RPE再会計・詳細科学レビューはユーザー／GPTへ戻す。

有限calibration一path・local binary64 evidenceであり、全law期待価格、新規性、他q・他cut・他compiler、RPE shot数／要求energy精度・最終全RPE優位、外部再現／immutable CIは資格付けていない。

## 証拠・公開・STOP

[PLAN](PLAN.md)／[実装・照合記録](IMPLEMENTATION_AND_CORRECTNESS.md)／[科学SOURCE](../../../artifacts/track_d_head_tail_h4/2026-11-08/run01/SOURCE.json)／[raw](../../../artifacts/track_d_head_tail_h4/2026-11-08/run01/RAW_CHECKS.json)／[原科学call数](../../../artifacts/track_d_head_tail_h4/2026-11-08/run01/CALL_COUNTS.json)／[全7資源](../../../artifacts/track_d_head_tail_h4/2026-11-08/run02_summary_completion/RESOURCES.json)／[全比較](../../../artifacts/track_d_head_tail_h4/2026-11-08/run02_summary_completion/COMPARISONS.json)／[最終集約](../../../artifacts/track_d_head_tail_h4/2026-11-08/run02_summary_completion/PAIRED_AND_CHI.json)／[要約](../../../artifacts/track_d_head_tail_h4/2026-11-08/run02_summary_completion/SUMMARY.json)／[原本hash索引](../../../artifacts/track_d_head_tail_h4/2026-11-08/run01/PAYLOAD_INDEX.json)／[最終STOP](../../../artifacts/track_d_head_tail_h4/2026-11-08/run02_summary_completion/STOP.json)。

元mainのHEAD/dirty diff/status、旧証拠45ファイルと旧run87ファイルは不変。events/QPY/NPZ/private入力・運用情報は公開しない。q64 controller/worker・SOURCE/output・CPU affinityや共有scheduler/venv/cacheへの操作は0。

通常git pushは環境のHTTPS認証情報不足で失敗したため、接続済みGitHub機能で同一treeをcommit/refへ保存する。結果公開後、remote SHA/tree・公開file bytesを照合して**mandatory STOP**。次の判断はユーザー／GPT。追加path、追加compile、C2、別研究へ進まない。
