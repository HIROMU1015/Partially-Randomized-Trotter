# 価格quantity・event単価・有限compile検証案

**今回、価格値・新compileは取得していない。** 以下は別途承認するための設計で、new lowering/modelを科学的採択したものではない。

推奨する最短経路は、既存P3の証拠を参照→A75fb0のoperator/dictionaryと必要なsignal検証→下記B0の実接続一pathである。案Aの20 prototype compileは角/価格定義を切り分けたい場合の選択肢で、B0の必須先行作業ではない。

## quantityと取得するmetrics

主指標はP0・測定付きfull controlled Hadamard wrapperの**全compiled RZ数**。Qiskit1.3.0/NumPy1.26.4、basis rz/sx/x/cx、opt1、seed17、backend/target/coupling/layout/routing追加指定なし。X/Yは個別取得し、YをXで代用しない。controlの実作用、scalar、測定範囲を揃える。Pauli routeとDF routeに同一のlowering文字列を強制しない。

補助指標はcompiled CX数、RZ/CX depth、total depth、size、qubits、必要ならangle分類。既存rte_compiled_costが返す`transpiled_circuit`とmetrics/fingerprint/CompilerSettingsを再利用し、別compilerを作らない。

**非Clifford RZ**は主RZと別metric。post-transpileの各RZ(theta)についてθのdecimal/hexを保持し、π/2整数倍からの距離を記録する。厳密にπの有理倍として与えられたangleなら、既存 `algorithm_codesign/df_correction_comparison/angles.py:clifford_sequence` のexact ruleを参照できる（θ=kπ/2ならClifford、π/4は非Clifford）。そのsynthesis/Gridsynthは呼ばない。

binary64角だけの場合の候補ルールは `d=abs(remainder(theta,pi/2))`、`d<=epsilon_C`を**許容差内Clifford診断**とする。例えばepsilon_C=1e-12 radを実行前に固定する案だが、今回採択・計算していない。εとdを併記し、厳密判定と区別する。角度を丸めたりCliffordへ置換しない。phase/global phaseを落とさず、単なる非Clifford RZ個数をfault-tolerant T価格にしない。人工行列testのatolはこのεとは別である。

元event.rotation_angleやprecompile gate listだけからcompiled非Clifford数を求めない。compilerはCRZ分解・1q融合・basis変換/相殺を行う。現metrics cacheの永続payloadは角列/QPYを保持しないため、cache hitでactual compiled circuit/角記録がないときは非Clifford数をnullにする。新規取得なら既存cost objectから一回走査してangle sidecar又はQPYを保存する小差分で足りる。保存QPYの新解析は別科学scopeである。

## c_Eとλをどう比較するか

単独event wrapperのC、empty/scalar capを差し引いたΔC、短いwordのC、head込みfull Cは別quantity。`C(A;B)−C(B)`はそのcompiler contextでの差であり、一般のevent単価/厳密な上下界ではない。`C(word)`を各eventの単独Cの和としない。反復Pauliの角加算、逆回転の相殺、parity/basis/DF-head境界は一体compileに反映される。gate fusionを先に仮定した数え上げでCを代替しない。

DF eventとPauli eventを**同一unitary**として比較できるのは、同じsupport/角/phaseへ写せるケース（例：identity-basisのZ/ZZ）だけ。一般のGaussian回転DF eventは複数Pauli項へ展開されるため、Pauli一eventと一対一対応しない。一般比較は同じ物理head/tail/targetをAが保証した上で、それぞれの辞書のlawとfull-word価格を比較する。方式間の別seed/cohortをpaired標本と呼ばず、同pathのS/CとX/Yの対応を保持する。

λが変わるとcomponent確率、tau=λ*piece_time/R、rotation angle、order確率/normalization、合法R配分・event数・B/Nが変わり得る。`G_R∝N c_E κ λ² T²`は影響を見るモデル式であり、c_E一定・N不変・価格加法性の証明ではない。新λ/finite-K signalはA、N/riskはBが担当する。Dで新実H4のλやN/Gを評価しない。

既存Cの加法モデル/単一full-wrapper残差のインターフェースは再利用できる。ただし元DF E[S]・残差係数/価格区間をPauliへ移さない。support weight、X/Y数、ordered products/隣接label、辞書law、head/outer境界等の新featureを使う場合も、その期待値/sourceとcompilerの検証が必要。source/hash一致だけでmodel誤差が0になるわけではない。

## 有限案A：短い固定wordのcompiler診断（最大20 wrapper）

**実分子を使わない、人工3 system wires＋ancilla3の価格benchmark案**。Aの辞書を推測で作る代わりにはしない。固定τ=.23、u=atan(τ)、scalar=.13、total T=.4を候補とし、同じ外側scalar capとmeasured X/Yを全wordへ付ける。prototypeは確率標本ではなく固定program、RNG/sampling0、compiler seed17。

このτは独立に指定する人工rotationパラメータで、headのT/qから実HamiltonianのλやRPE lawを定義するものではない。

| program | 対象・選定理由 |
|---|---|
| E0 | empty evolution＋scalar cap。wrapper固定費の参照 |
| PZ / DZ | Pauli IIZ、DF support(0)、identity basis、同じ+u回転。単一Zの同unitary比較 |
| PZZ / DZZ | Pauli IZZ、DF support(0,1)、identity basis、同じ−u回転。新DF ZZとD2の同unitary比較 |
| PXYZ | Pauli XYZ、+u。X/Y局所basisとweight3 parityの例 |
| PK2 | K2、rotation XYZ負係数、products ZII→YII正係数、event phase−1。非可換product/相対phaseを含む |
| PREPEAT | 同じPZZを2回、元順序のまま連結。同角反復・融合の診断 |
| H0 / HP | 同じ人工DF one-body＋一fragmentのhead、q2/T.4。H0はempty tail、HPはouter0にPK2・outer1にPZ。head-tail/outer境界の診断 |

計10 controlled body、各X/Yで**新wrapper20・新compile最大20**。E0/PZ等もfull-wrapperなので状態準備費を含めず、scalar/measurementを含める。各programは1固定word、trajectory mean/CIではない。H0/HPは同じ人工headを使用し、既存native testの2-mode blockをzeroの第3modeで拡張する案。実H4のheadを生成しない。

PK2の明示入力案はordered components `(XYZ,-.3),(ZII,+.1),(YII,+.1)`、人工probabilities(.6,.2,.2)、finite_rte_distribution(.23,2)、_make_eventのindices(0,1,2)/order_index1。rotation角はatan(.23/3)、product順ZII→YII、order phase−1を元関数で生成する。実辞書/λの値を仮定したものではない。

PauliはP3のappend_event＋D2、DF比較primitiveは元DF event builder＋新controlled_ZZ、headはnative controlled_tails API、wrapper/compileは元pipelineを使う。K2は元_make_eventへ明示indexを渡し、forced normal sampleと偽装しない。cohort/tag案は `D_DF_HEAD_PAULI_PRIMITIVES_V1`、program名/index0、axes XY、input/source/angle hexを固定する。sampling seedはなくseed17はcompiler専用。

価格取得前に全16×16 controlled行列（またはその小さいblock）を独立Pauli/Fock oracleへ照合し、low/high control・global/relative phase・scalar一回・X/Y符号・測定を確認する。今回の9人工testは接続原理の証拠であり、この未buildの10 programを価格取得済み/全PASSとはしていない。

これで分かるのは特定wordのC/角分類/CXと非加法性だけ。weight4〜8等の実Pauli tailを網羅せず、支持頻度・λを持たないので実辞書の平均c_Eを認定しない。実辞書に重要なsupport/basis/product classがあるならA descriptorを受けて**必要なclassと有限数を改めて固定**する。10 prototypeを最終探索空間の制限にしない。

## 有限案B0：Aに合わせた最初の実接続（最大2 compile、未実行）

A75fb0の条件付き案に合わせ、保存H4 DF10・元head prefix1/one-body、S2/ΔPF=.1/h=.1/M12/K2/元exactκの**LD1・m0・T=.1・qPF=1**を第一候補とする。Pauli側Rは新λと元mapperに依存し、まだ未知なので指定済みと偽装しない。

fresh normal1 path/index0、cohort案 `D_DF_HEAD_PAULI_LD1_M0_V1`、X/Y両軸、evolution1・wrapper2・cache missでcompile最大2。R・辞書付きcanonical/physical recipe・新SOURCE・seed family/全seed・元input/preparation/scalar値をA/Bの固定descriptor後に決める。SOURCE/数値条件が揃うまではprepareもsamplingもしない。

元native DF head＋Pauli D2 tail→measured native wrapper→既存compiler/cacheを使い、SOURCE束縛、同一event列の正しいcontrolled作用、scalar、X/Y、metrics/angle保存とwall/RSSを確認する。必要な実入力復元1回、head/Pauli context構築1回、normal trajectory1回、full native/operator・両wrapper/compiled作用検査をcaller SOURCEに明示する。full256を使い、sector射影eventや新reference/state solveを追加しない。

これはC/S/mean modelの適格性を認めるpilotではない。Sが未対応ならnull。既存DF LD1/m0 P1/P2のdirectはSOURCE/元lawの参照に使い、ただ比較相手を作るための再compileはしない。新候補への受入/価格差の数値再解析が必要なら、その別科学scopeを含めて承認を受ける。異なる辞書pathの単回差を平均単価差にしない。

## 有限案B1：必要になった場合だけの平均設計pilot

案Aを必須の先行20 compileとしない。Aが同じ物理head/tail/targetの合法descriptorを公開したら、実際の必要classを使う短い一cellへ直接絞れる。

初期予算の候補はPauli辞書・DF辞書それぞれ**fresh normal2 path/index0,1・X/Y**、計4 path/最大8 compile。一cellのT/q/R/K/λ/scalar・physical IDはAの合法scheduleを採用し、現在は未指定。2本はpilot内変動を初めて観測できる最小数という設計で、平均c_E差の精度/CIを満たす数ではない。

cohort案は `D_DF_HEAD_PAULI_SHORT_PILOT_P_V1` と `D_DF_HEAD_PAULI_SHORT_PILOT_DF_V1`。使用する既存seed family・全整数seedはAのidentity/生成SOURCEが揃ってから凍結し、未定seedで実行しない。別辞書を同じindexでpaired扱いせず、X/Yは同じpathを共有。forced rareを追加しない。n増加や全roundへ自動移行しない。

単一eventのE[c_E]を推定するなら、eventが出る位置/角度/lawとwrapper referenceのquantityを事前に定義する。短word総Cから割った値はそのword/lawに依存する有効単価であり、一律係数ではない。必要精度は比較したい価格差と最終的なN/λ/候補familyからC/Bと決める。旧P3/O1のSEやnを新条件へ代入せず、law/rare/共有誤差がUNKNOWNなら科学的順位は保留する。

## SOURCE・metrics・STOPと承認範囲

案A/B0/B1は独立の候補で、20＋2＋8 compileをまとめた実行menuとして採択しない。B0のpathを将来のB1統計へ自動poolもしない。実行caller SOURCEは未作成。既存D3d39、P3 native9c4006bf/provider c31を基に最小接続だけを人工確認し、完全commit SHA・program又はcandidate/round・axes・path/cohort/全seed・compile数・出力先・angle分類ε・作用検査callを固定してから明示承認を受ける。

新出力先案は `docs/research/d_df_head_pauli_price_execution_20261114/<scope>/run01/`、privateは `/home/AbeHiromu/projects/d-df-head-pauli-price-20261114-private/<scope>/run01/`。旧P3/O1/T2の出力・grant・STOPを再利用しない。compileには同じprocessで同時transpileしない既存process分離を使い、必要並列度は実資源/独立taskで選ぶ。CPU/RAM/workerに恣意的な低いcapを追加せず、共有設定や他者に干渉しない。

取得するのは各axisの全RZ・非Clifford診断RZとangle列/ε・CX/depth/size、native/compiled fingerprint、compiler/native/lowering・辞書/入力/preparation/scalar SOURCE、event/order/seed、cache、wall/compile時間/peak RSS。Sが適合未確認ならnull。旧receiptの差し替え・histogramからの非Clifford推測はしない。

SOURCE/input/辞書/位相/角規約/作用gate/compiler不一致、原本欠測、継続不能では部分結果を保存してSTOP。tolerance緩和・phase alignment・generic fallback・別seed/round/retryはしない。終了時にもsample数増加、fit、new λ/signal/N/G、model/lowering採択、長時間外挿は範囲外。現在は調査・人工test・計画の公開照合後 **mandatory STOP**。
