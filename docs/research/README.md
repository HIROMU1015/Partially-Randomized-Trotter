# 研究計画資料

## 研究C：第三γ0 DF・修正後493-call完了、switch転移判定不成立（2026-10-29）

[結果・32 full512・holdout・条件付き費用](research_c_df_basis_third_repair_execution_20261029.md)。保存直鎖H4/1Å/STO-3G、通常γ0 CDF8/元順LD2、partial S2 δ=h=.1/s1/r128/K0・13round、own-ground/状態準備除外。Components metadataだけを修正し人工46 PASS、SOURCE30436b142を固定してfresh run02で493/493 COMPLETE、correctness FAIL0。Raw/compiled/actual adjoint/X/Yは各32 PASS、QPY128。RZ holdout MAEはA369.78698、B303.32856、mean2892.23438のためSWITCH_TRANSFER_NOT_SUPPORTED。Kernel平均RZ17454.1875、ideal仮定shots21754/総RZ627057959562.5（M4比2.678387、P4比2.276714）。有限数値pilotで全law/actual CA/cost support/因果・汎化/η採択は未資格。旧SOURCE112/run01/保護1725件不変、新fit0、技術relaunch1/run内retry-resume0。公開照合後mandatory STOP。以下の未完了/旧STOP表記は各時点の履歴。


## 研究C：第三γ0 DF・141-callでnative metadata技術STOP（2026-10-29）

[部分結果・全13round・STOP原因](research_c_df_basis_third_execution_20261029.md)。保存H4/1Å/STO-3G、γ0 CDF8/LD2、partial S2 δ=h=.1/s1/r128/K0、own-ground/状態準備除外。SOURCE14270239a無変更で141 attempts=140 COMPLETE+1 technical FAIL。表現/ground/alias・13/13 PF-RTE十分条件PASS、ideal仮定shots21754。Native full_law_gateへ渡すmetadataのcomponents欠損でSTOPし、sampling/build/compile/DF holdout0。第三DF価格/予測誤差/総費用はnull、旧C不適格不変。SOURCE112/input93/旧保護1667件照合、新fit/retry/resume0、旧原本保全。公開照合後mandatory STOP。以下の未実行表記は当時の履歴。

## 研究C：basis-switch-aware設計・保存γ0第三DFの準備（2026-10-29）

[数式・再利用・人工検証・未承認493-call案](research_c_df_basis_switch_optimization_design_20261029.md)。固定exact-key partitionのswitch項はJのmass比に依存し、自由Uの幾何を直接最適化しない。Exact/smooth零tail・ε bias・非凸性・gaugeを分離し、η/新目的は未採択。保存H4/1Å/STO-3G、γ0 CDF8/LD2を第三候補としてown h=.1/s1/r128/K0の13round、条件付きRPE、GO時native32、DF holdoutへ接続。旧2cohortの共通interceptを第三価格前に固定し、Cは復活させない。新21＋回帰110＝131 PASS、科学call0、SOURCE112/input93/旧原本1667件照合、旧103 SOURCE/STOP不変。別科学許可までmandatory STOP。

## 研究C：basis-key修正後148-call完了・複合surrogate未資格（2026-10-29）

[結果・モデル比較・旧STOP保全](research_c_df_basis_surrogate_repair_execution_20261029.md)。保存直鎖H4・1Å/STO-3G、CDF8/LD2、partial S2 δ=h=.1/s1/r128/K0、M4 Proposed/tail-P4各32件。既存spin_liftによる厳密key修正、新7＋回帰43＝50 PASS、SOURCE030128649…を固定しfresh run02で148/148 COMPLETE、correctness FAIL0。RZ MAEはM4でA221.41<C227.80、P4でB261.54<C273.58のため両cohortともSURROGATE_NOT_QUALIFIED。全192 fold full rank・負予測0。静的DFは2個、因果/新DF汎化/η採択は未資格。新fit/sampling/build/compile0、旧SOURCE99/run01/保護1570件不変。技術relaunch1、run内retry/resume0、公開照合後mandatory STOP。以下の未実施/旧STOP表記は履歴。

## 研究C：basis-aware saved64・3-callで技術STOP（2026-10-29）

[部分証拠・原因候補・残る修正](research_c_df_basis_surrogate_execution_20261029.md)。保存直鎖H4・1Å/STO-3G、CDF8/LD2、PR s1/r128/K0の2候補各32件を対象に固定SOURCEb1985ff01で開始。3 attempts=2 COMPLETE+1 FAIL、M4_Proposed/static_featuresのEXACT_SAVED_BASIS_KEYでSTOP。Trajectory/QPY/model未実施、性能未判定。既存spin_liftと新np.kron経路のsigned-zero repr差を人工2×2だけで再現。SOURCE/契約変更・retry/resume0、旧1458件保全、公開照合後mandatory STOP。

## 研究C：basis-aware native surrogateの技術設計（2026-10-29）

[3案・実装対応・2cohort schema・未承認148-call案](research_c_df_basis_native_cost_surrogate_design_20261029.md)。切替/overlap/複合run-atomを比較し、複合特徴ベクトルを推奨。新12＋回帰31＝43人工PASS。保存H4の新特徴量・相関・NPZ/QPY数値decode・fit/sampling/build/compileは0。η/scale/新目的は未採択。2静的DFを64標本扱いせず、次は保存各32件の独立cohort内記述検証だけを別許可で実行。旧原本保全、公開照合後mandatory STOP。

## 研究C：tail-P4 PRの449-call完了・総資源改善なし（2026-10-28）

[PF/RTE・shots・損益分岐・native・総資源](research_c_df_structural_pr_performance_20261028.md)。保存直鎖H4・1Å/STO-3G、tail-P4 CDF8/LD2、partial S2 δ.1/s1/r128/K0/m0..12、own-DF oracle/状態準備除外。SOURCE4ac0e5f96無変更、449/449 COMPLETE、correctness FAIL/retry/resume0。全13round適格、主仮定shots16,568、fixed32/full512/adjoint/X/Y PASS。Kernel平均RZ15,047.46875、M4-r128比1.068983。条件付き総RZ275,422,425,481.5/旧234,117,788,135=1.176427。疎化による価格/総資源改善は今回の点比較では観測なし。全law/actual CA/母平均優位は未資格。旧1054件保全、公開照合後mandatory STOP。

## 研究C：tail-P4のPR・条件付きnative32経路（2026-10-28）

[再利用・40人工PASS・449-call段階付き契約](research_c_df_structural_pr_preparation_20261028.md)。候補固有role、保存J/U固定、new13round/ground/law、登録GO時だけnative32。科学開始前SOURCE固定、旧gate/source不変、未知全law誤差はnull。

## 研究C：tail J固定support CDFの1 fit・DF予算PASS（2026-10-28）

[結果・SOURCE・call・STOP](research_c_df_structural_support_execution_20261028.md)。保存直鎖H4・1Å/STO-3G、CDF8/LD2、head-full/tail-P4、γ.1。SOURCE59e186ad無変更、21/21 COMPLETE、1 fit、retry/resume0、correctness FAIL0。終了sector誤差7.45156e-9Ha＋guard3.26388e-9HaでCA/8 PASS。Tail係数位置144だがλ1.51617Haは旧PR-aware1.44243Haより大きく、性能GOではない。PF/RTE/shots/native/総費用0、旧1006件保全、公開照合後mandatory STOP。

## 研究C：DF構造改良の技術提案・科学実行未承認（2026-10-28）

[横断調査・3案・再利用地図・最小契約](research_c_df_structural_design_20261028.md)。保存H4・1Å/STO-3G、CDF8/LD2を維持し、既存γ/r探索を拡張せずtail Jの固定support制約を最初の検証案とした。既知support制約CDFの限定拡張で独立新規性/資源優位は未認定。旧loss/AD/codec/SciPyを再利用するcompact adapter、1-fit限定runner、人工60 tests＋29 subtests PASS。未承認21-callはDF評価だけで、実H4decode/fit/PF-RTE/shots/native0。後続は新J/U固有の別契約が必要。旧SOURCE/raw/STOP保全、remote照合後mandatory STOP。

## 2026-10-27：固定F2 polar・κの厳密Hermiticity条件でSTOP

[結果・partial evidence](research_c_pr_aware_df_fixed_F2_polar_execution_20261027.md)。保存直鎖H4・1Å/STO-3G、通常number CDF8 Baseline baseline_00、元順LD2、partial S2 δ.1/s4/h1/40/r16/K0、sector36/full512、own-DF oracle/状態準備除外。Right-polar CAR targetを今回限定採択し、SOURCE45ae98943492e921c22a0f527d5587cd1e06462f／新契約を固定して実行。4/最大7 attempts＝3 COMPLETE＋1 FAIL、残3 NOT_RUN。CAR_reference入口の元g補正κがEXACT_HERMITIAN_INPUT_REQUIREDでSTOP。Raw/CAR・bridge・triangleは未評価/nullで、raw回路の測定correctness FAILではない。追加診断・κ変更・retry/resume・F3なし。旧B1/F1/C1の資格とraw/STOP不変、公開照合後mandatory STOP。

## 2026-10-27：固定Proposed F1・全基底validated評価

[結果・資格](research_c_pr_aware_df_fixed_proposed_f1_execution_20261027.md)。保存H4・直鎖1Å/STO-3G、通常number CDF8 proposed_00、元順LD2、partial S2 δ.1/s4/h1/40/r16/K0、system8+last ancilla/full512、own-DF oracle/状態準備除外。SOURCE `e7d22a404442234662ef6d117c55dd9868217c9e` を無変更で一度実行し7/7 calls、PASS3/correctness FAIL0/UNRESOLVED0/NOT_RUN0、retry/resume0。Arb192全512基底・位相込みの有限固定回路保証。共通shell6 certificateはidentity一致で再利用。Raw/CAR・全law・CA/保証shots/総資源は未資格、F2 executable=falseを維持。

## 2026-10-27：固定CAR reference・Proposed検証の技術準備

[技術報告・数学的target・210人工回帰PASS](research_c_pr_aware_df_fixed_car_reference_readiness_20261027.md)／[F1未承認7 calls](research_c_pr_aware_df_fixed_proposed_contract_draft_20261027.json)／[F2 executable=false draft](research_c_pr_aware_df_fixed_car_contract_draft_20261027.json)。literal CARのArb評価とanalytic polar bridgeを実装したが理想targetは未採択。旧B1・C1資格を変更せず、実H4科学call0、mandatory STOP。

## 2026-10-27：C2-A B1・保存固定回路の事後validated評価

[実行報告](research_c_pr_aware_df_c2a_fixed_b1_execution_20261027.md)。固定SOURCE `518a730510a57d0cafa1575a91f5f1979c203be3`を無変更で一度実行。保存H4・1Å/STO-3G、通常CDF8 baseline_00、元順LD2、partial S2 δ.1/s4/r16/K0、full512、own-oracle/状態準備除外。19/19 units、9判定はPASS9/FAIL0/UNRESOLVED0/NOT_RUN0、retry/resume0。Arb192全基底・位相保持の固定gate列証拠であり、raw-Hamiltonian/reference・全law・CA/保証shots/総資源は未資格/null。新fit/build-compile/sampling/C1再解析0、旧SOURCE/raw/STOPとC1資格は不変。公開照合後mandatory STOP。

## 2026-10-27：C2-A全lawの成立可能性・保証方針比較

[A/B比較・技術報告](research_c_pr_aware_df_c2a_guarantee_strategy_readiness_20261027.md)／[保存1kernel＋3shell・未承認19-call B1契約](research_c_pr_aware_df_c2a_guarantee_contract_draft_20261027.json)。RZ/ZSXX全入力保証は未取得、現証拠でA-3。B1の有限全基底Arb検証を人工実装。旧all-law blocked契約・C1約83%仮定付き点推定は変更しない。新37＋旧193 PASS、旧不安定1件deselect、実H4科学call0。公開照合後mandatory STOP。

## 2026-10-27：ZSXX全到達domain・条件付きwhole-P0

[技術報告・Q1〜Q5](research_c_pr_aware_df_zsxx_uniform_domain_readiness_20261027.md)／[新しい実行不可契約](research_c_pr_aware_df_zsxx_uniform_contract_draft_20261027.json)。RZ/SX/Xのdet4 coset、固定kernel grammarの条件付きN≤67731、逐次run積／true-period phase／whole-P0式、CRZ等の部分規則を準備。Rust uniform imageと全translationは未資格。新規32＋旧161 PASS、旧順序依存1件deselect。実H4科学call0、公開照合後mandatory STOP。

## 2026-10-27：C2-A・ZSXX captured-pair validated enclosure

[技術報告・Q1〜Q5](research_c_pr_aware_df_c2a_zsxx_enclosure_readiness_20261027.md)／[未承認・実行不可契約](research_c_pr_aware_df_c2a_zsxx_contract_draft_20261027.json)。固定Rust出力をArb192で囲う部分的成果。人工41例＋whole-P0/adjoint各2例、新規52＋回帰109 PASS、旧順序依存1件deselect。全law／一様initial translation／Rust数学関数モデルは未資格。科学call上限null・実H4科学call0。旧SOURCE・証拠保全、公開照合後mandatory STOP。

## 2026-10-27：PR-aware DF C2-A・whole-P0構造と未資格上界の技術準備

[固定ZSXX graph・条件付き回数制限・人工/回帰110 PASS](research_c_pr_aware_df_c2a_synthesis_readiness_20261027.md)／[実行不可の未承認草案](research_c_pr_aware_df_c2a_contract_draft_20261027.json)。新`c_pr_aware_df_c2a.py`／passive-only `check_research_c_pr_aware_df_c2a.py`／同名management test。Basis閉包等の前提下で採用rewrite≤N・採用matrix入力≤6Nを構成。Uniform ZSXX/initial translation/reference enclosureは未資格、科学call上限null・executable=false。SOURCE93/input23と固定Qiskit14SOURCEをpassive確認、新実H4科学call0、旧証拠保全、公開照合後mandatory STOP。


## 2026-10-27：PR-aware DF C2-Bridge・47-call数値対応評価完了

[全gate・26round・未資格事項](research_c_pr_aware_df_stage_c2_bridge_execution_20261027.md)。固定SOURCE3f64842d69d54a70a5b736486b5faefa398903b8を無変更で一度実行し47/47 COMPLETE、correctness FAIL/retry/resume0。保存直鎖H4・1Å/STO-3G、通常number CDF8 Baseline/Proposed、Nα=Nβ2/sector36、元順LD2、partial S2 δ.1/s4/h1/40/r16/K0/m0..12、own-DF oracle/状態準備除外。両216 atom/K0 mean/保存CAR kernel gateと26round replayを確認。観測operator2差の最大はBaseline6.53148e-11（m11）/Proposed9.46816e-11（m12）、signal差の最大は5.98984e-11/8.10860e-11。数値対応pilotであり、厳密Bridge/reference enclosure・全law P0・PCG64 IID・CA/保証shots/totalは未資格/null。C1の約83%点推定不変、新fit/sampling/build-compile/DF-PF再計算/shots/総費用/C1再解析0。SOURCE82/input20・旧1799件不変、raw44と全26round CSVを保存、公開照合後mandatory STOP。


## 2026-10-27：PR-aware DF C2・Reference Bridge統合準備

[Reference対応・P0 coverage・人工/回帰85 PASS](research_c_pr_aware_df_stage_c2_bridge_synthesis_readiness_20261027.md)／[未承認47-call Bridge契約](research_c_pr_aware_df_stage_c2_bridge_contract_draft_20261027.json)。SOURCE82/input20のpassive確認、実H4科学call0。全law synthesis/reference enclosureは未資格、C1保証shots/totalはnullのまま。旧証拠保全、別途明示科学許可までmandatory STOP。

## 2026-10-27：PR-aware DF C2-B・35-call確率表解析完了

[全gate・2表／26round・資格の限界](research_c_pr_aware_df_stage_c2b_execution_20261027.md)。固定SOURCE3c1bb2d9c…を無変更で一度実行し35/35 COMPLETE、correctness FAIL/retry/resume0。保存H4・1Å/STO-3G、通常CDF8二arm、sector36、元順LD2、partial S2 δ.1/s4/r16/K0/m0..12、own-DF oracle／状態準備除外、共通Gaussian/P0。各216 atomの既存make_tail再生PASS。IID一様53-bit格子モデルのTVはBaseline3.6470735282e-15／Proposed3.2261762373e-15、m12条件付きraw信号上界1.9121168860e-9／1.6914454871e-9。PCG64 IID、CAR/atom/angle/normalization bridge、全law P0 synthesis、support、actual CA／保証shots／totalは未資格/null。新fit/sampling/build-compile/operator/shots・総費用・C1再解析0、SOURCE76/input12・旧1768ファイル保全。公開照合後mandatory STOP。以下は当時のscopeの履歴。


## 2026-10-27：PR-aware DF C2・全law資格の技術調査／未承認C2-B準備

[NumPy/P0 SOURCE・資格の限界・人工99 PASS](research_c_pr_aware_df_stage_c2_qualification_readiness_20261027.md)／[未承認35-call表解析契約](research_c_pr_aware_df_stage_c2_contract_draft_20261027.json)。`c_pr_aware_df_stage_c2.py`／静的`c_pr_aware_df_stage_c2_audit.py`／`run_research_c_pr_aware_df_stage_c2.py`／同名management test。Exact finite-grid TVと条件付きraw-law伝播、scoped synthesis／support入力、null C1-previewを人工検証。現whole-P0合成・PCG64 IID／CAR-atom-reference bridge・費用supportは未資格。SOURCE76/input12 passive PASS、新規実分子科学call0、C1再解析0。保存H4・CDF8二arm・元順LD2・δ.1/s4/r16/K0/m0..12・own-oracle/状態準備除外を維持。別科学許可後のみC2-B実行、公開照合後mandatory STOP。


## 2026-10-27：PR-aware DF C1・仮定付きshots／serial総資源評価完了

[164-call結果・全78round・資格の限界](research_c_pr_aware_df_stage_c1_execution_20261027.md)。固定SOURCE b1c735da3…を無変更で一度実行し164/164 COMPLETE、correctness FAIL/retry/resume0。保存H4・1Å/STO-3G、通常CDF8 Baseline/Proposed、sector36、元順LD2、partial S2 δ.1/s4/r16/K0/m0..12、own-DF oracle／状態準備除外、共通precision/P0／CA.00159／serial model。証拠経路26roundはUNRESOLVED・shots/total null。2つの明示仮定scenarioでは両armが条件付き十分条件内。Idealのshots53136/19574、総RZ点推定3,194,333,274,946/540,290,160,109（Proposed約83.086%低い）。.001Aをsynthesis/law各項へ仮定するscenarioも約83.089%低い。全law誤差・費用support・actual CA・保証shots/優位は未資格/null。SOURCE60/input408・既存1644ファイル不変、新fit/sampling/build-compile/量子shots0。C2は未実行・次指示待ち、公開照合後mandatory STOP。以下は以前のscopeの履歴。


## 2026-10-27：PR-aware DF Stage C・保存証拠からshots／serial資源への技術準備

[2経路・資格不足・人工73 PASS・C1/C2設計](research_c_pr_aware_df_stage_c_readiness_20261027.md)／[未承認C1・164-call案](research_c_pr_aware_df_stage_c1_contract_draft_20261027.json)。新`c_pr_aware_df_stage_c.py`／専用runner／人工test。Evidence laneは全law合成・実装law不足でnull、2明示scenarioで既存accuracy/Hoeffding/serial-costへ接続する案。SOURCE60/saved408のpassive照合PASS、旧1361ファイル不変。実H4新shots/総費用/再解析・新科学call0。C2はC1後の選択を要する非実行草案。公開照合後mandatory STOP。

## 2026-10-27：PR-aware DF B2・64 native kernel校正完了

[全gate・費用6指標・不確かさ・技術再実行](research_c_pr_aware_df_stage_b2_execution_20261027.md)／[修正契約](research_c_pr_aware_df_stage_b2_compat_contract_20261027.json)。保存H4・1Å/STO-3G、CDF8 Baseline/Proposed、元順LD2、partial S2 δ.1/s4/h1/40/r16/K0、own-DF oracle/状態準備除外、共通precision/P0。Run01は429 unitsでQiskit Already borrowed技術STOP（数値FAIL0）を保全し、共有build/compile/adjoint mutex・元64seedのfresh run02が652/652 COMPLETE／FAIL0、累積1081、技術再起動1・run02 retry/resume0。64 full512/adjoint/X-Yと67 compileがPASS。RZ標本mean±SEは5198.3125±66.7146対4765.875±80.3845（今回約8.3188%低い）。母平均保証・support・CA/shots/総資源は未資格。SOURCE50/input22/旧628不変、公開照合後mandatory STOP。

## 2026-10-27：PR-aware DF B2・native費用校正の技術準備

[共通native・full512・人工72 PASS・費用資格と制限](research_c_pr_aware_df_stage_b2_preparation_20261027.md)／[未承認652-call契約](research_c_pr_aware_df_stage_b2_contract_draft_20261027.json)。新`c_pr_aware_df_stage_b2.py`／専用runner／人工test。32独立trajectory/arm、64 kernel＋3shell、67 P0 compile、全operator/adjoint/X-Y後にwhole-kernel metricsを記録する案。SOURCE47/input22のbyte/header preflight PASS、旧175ファイル不変。実H4 decode/sampling/build-compile/cost・新科学call0、CI/support/ideal費用/CA/総資源は未資格。公開照合後mandatory STOP。

## 2026-10-27：PR-aware DF B1・両armの13/13 roundが潜在適格

[全gate・26セル・信号・call・資格の限界](research_c_pr_aware_df_stage_b1_execution_20261027.md)。固定SOURCE7da0fdb3f…無変更、141/141 COMPLETE、correctness FAIL/中断/retry/resume0。保存H4・1Å/STO-3G、sector36、Stage A最終CDF8 Baseline/Proposed、元順LD2、partial S2 δ.1/s4/r16/K0/m0..12、own-DF ground oracle・状態準備除外。両POTENTIALLY_FEASIBLE。m12のAは.209598/.513867、補正RTE誤差.164305/.125612、PF.00447695/.00447625。newfit/DF norm/native/build-compile/sampling/shots/cost0、全law合成・実装law・CA・総資源未資格。旧証拠保全・公開照合後mandatory STOP。

## 2026-10-27：PR-aware DF Stage Bの接続・人工検証準備

[2 CDF arm／law／sector reference／共通native接続・人工56 PASS](research_c_pr_aware_df_stage_b_readiness_20261027.md)／[未承認B1-only141-call案](research_c_pr_aware_df_stage_b_contract_draft_20261027.json)。新`c_pr_aware_df_stage_b.py`／`stage_b_runner.py`／同名runner・management test。保存Stage A J/Uの新評価は0、hash/header/JSON preflightのみPASS、旧97ファイル不変。B1は2arm×13roundのPF/RTE/A/signalをsector36で検証し、build/compile/sampling/shots/costは0の案。B2は人工native部品まで、実行範囲は別固定。科学call0、公開照合後mandatory STOP。

## 2026-10-27：PR-aware DF Stage A V2・物理予算とexact tail mass基準がPASS

[全gate・paired fit・gradient・科学call・限界](research_c_pr_aware_df_stage_a_x0_v2_execution_20261027.md)。固定SOURCEe1085c95f…無変更、30/30 COMPLETE、correctness FAIL0、各1 fit計2、retry/resume0（旧18含む累積48）。保存H4・1Å/STO-3G、sector36、通常CDF8、元順prefix LD2、共通初期値、γ0/.1。両終了DF norm＋guard1.02976e−8/1.22956e−8 Haで予算内、exact tail mass2.20976184/1.44242989 HaでProposed34.7246%削減、STAGE_A_PROXY_AND_DF_PASS。相対目的減少停止でgradient収束・PR費用・最終CAは未資格、Stage B/Cなし。旧証拠保全・公開照合後mandatory STOP。

## 2026-10-27：PR-aware DFのx0接続を技術修正

[単一qualified x0引渡し・人工61 tests＋29 subtests](research_c_pr_aware_df_x0_repair_20261027.md)／[新V2・未承認30-call契約](research_c_pr_aware_df_pilot_contract_x0_v2_unapproved_20261027.json)。既存corrected-codecへopt-in prepared初期値を接続し、fit内の再logを除去。旧default・目的／gradient／optimizer条件は維持。実分子decode・新科学call0、旧18-call STOPを保全、新2026-10-27/run01未作成。別明示許可までmandatory STOP。

## 2026-10-27：PR-aware DF Stage A・optimizer前の技術的STOP

[初期予算・18-call部分証拠・x0接続](research_c_pr_aware_df_stage_a_execution_20261027.md)。固定SOURCE94d1c7b4b…無変更、18/30 units（16 COMPLETE／2 FAIL）、ffsim entry各1・実optimizer0・完了fit0・retry/resume0。保存H4・1Å/STO-3G、sector36、CDF8元順prefix LD2、γ0/.1。初期general-K/raw-CARとDF予算はPASS、再pack後の`QUALIFIED_X0_CHANGED`で両arm STOP。終了DF・exact mass削減・PR資源効果は未資格。旧SOURCE/raw/STOPを保全し、公開照合後mandatory STOP。以下の技術準備状態は履歴。

## 研究C：PR-aware DFの初期設計・技術準備（依頼識別子20261026）

[設計・再利用・人工49 tests＋29 subtests](<../../docs/research/research_c_pr_aware_df_optimization_design_20261026.md>)／[未承認Stage A・30-call案](<../../docs/research/research_c_pr_aware_df_pilot_contract_draft_20261026.json>)。`c_pr_aware_df.py`／`c_pr_aware_df_pilot.py`、`run_research_c_pr_aware_df.py`、同名management tests。既存ffsim loss/gradient・修正codec・SectorCARへ固定tail mass代理目的を最小接続。実optimizerは人工mock nit0、両env byte preflight PASS。実分子decode・新科学call0、科学的有効性／新規性・Stage B/Cは未資格・未承認、mandatory STOP。

## C1：H4 XDF9費用screen・両sでstrict改善を除外（2026-10-26）

[全gate・metrics・shots・比較・制約](research_c_c1_h4_xdf9_cost_screen_execution_20261026.md)／[raw48・QPY4・83-call journal・STOP・CSV](../../artifacts/research_c_c1_h4_xdf9_cost_screen/2026-10-26/)。固定SOURCE7328ecc9a…無変更、83/83 COMPLETE、correctness FAIL/inflight/retry/resume0。直鎖H4・1Å/STO-3G、Nα=Nβ2、元順XDF9、全frame S2 s1/s4、δ.1/m0..12、own-ground/preparation excluded、同precision/P0/shared/fenced serial。kernel RZ15720/各s。楽観的総RZ s1=144805110560/s4=530107748679はいずれも保存XDF8/CDF8より大きく、固定十分処方・実装・数値資格内でstrict改善を除外。full operator/actual合成誤差/CA/保証付きshots/最終費用は未資格・未評価。旧結果保全、公開照合後mandatory STOP。

## C1：H4 XDF9 native／楽観的shots screenの技術準備（2026-10-26）

[最小接続・単調性／ceil・限界・full512後続案](research_c_c1_h4_xdf9_cost_screen_preparation_20261026.md)／[未承認83-call契約](research_c_c1_h4_xdf9_cost_screen_contract_unapproved_20261026.json)／[SOURCE・人工／回帰97 PASS・passive証拠](../../artifacts/research_c_c1_h4_xdf9_cost_screen/2026-10-26/technical/)。新entry `run_research_c_c1_h4_xdf9_cost_screen.py`／module `c1_h4_xdf9_cost_screen.py`／同名management test。既存PrecisionBuilder/P0/solver/serial/rational楽観集計を再利用、XDF9 S2 s1/s4の2 kernelとPF26から固定処方の楽観的RZ下限を接続。実H4のdecode/build-compile/shots/cost・科学call0。下限は固定処方／serial実装内の数値screen、full operator／strict enclosure／actual CA・shots・資源優位ではない。SOURCE56/input24 byte preflight PASS、旧証拠不変、別明示許可までmandatory STOP。

## C1：H4 XDF9・表現／ground-alias／PF26がPASS（2026-10-26）

[全gate・全26 PF値・科学call・資格の限界](research_c_c1_h4_xdf9_pf_execution_20261026.md)／[raw45・SOURCE・journal・STOP・CSV](../../artifacts/research_c_c1_h4_xdf9_pf/2026-10-26/)。固定SOURCE9a4cda44a…無変更、95/95 COMPLETE、correctness FAIL/inflight/retry/resume0。直鎖H4・1Å・STO-3G、Nα=Nβ2/sector36、保存OpenFermion元順prefix9、元g補正κ・核scalar・frame順、全frame S2 s1/s4、δ.1・m0..12、PRなし、own-XDF9 oracle/状態準備除外。保存DF予算PASSを再利用、表現10因子・sum・own-ground/alias PASS、両sとも13/13 PASSのPOTENTIALLY_FEASIBLE。最大PF誤差s1 .06843041535127227/s4 .004277093283650178。新fit/DF norm/native/合成/CA/shots/総費用0、27 protected filesと原packet不変。旧約27%／約12%比較を保全し、公開照合後mandatory STOP。以下は過去の準備・結果の履歴。

## C1：H2／旧H4／新H4／H6統合と次期XDF9準備（2026-10-26）

[保存16候補・baseline差分・SOURCE来歴](research_c_c1_h2_h4_h6_integration_update_20261026.md)／[A:H4 XDF9・B:H6 CDF13の比較と未採択推奨](research_c_c1_next_minimal_experiment_selection_20261026.md)／[人工78 tests＋29 subtests・技術準備](research_c_c1_h4_xdf9_preparation_20261026.md)。H4約27%→約12%をrank単独の因果効果とせず、3系をサイズscaling実験と扱わない。新entry `run_research_c_c1_h4_xdf9_pf.py`／module `c1_h4_xdf9_pf.py`／同名management testは、既存CAR／一般K／PFを再利用する[未承認95-call契約](research_c_c1_h4_xdf9_pf_contract_unapproved_20261026.json)の準備。original input／SOURCE24／保存JSON3のbyte preflight PASS、実分子decode／新科学call0。旧証拠保全、推奨は未採択、公開照合後mandatory STOP。後続native／CA／shots／費用は別範囲。 [CSV・来歴](../../artifacts/research_c_c1_integration_update/2026-10-26/)／[SOURCE・人工log・passive証拠](../../artifacts/research_c_c1_h4_xdf9_pf/2026-10-26/technical/)。

## C1：H4 CDF8/XDF8の共通CA・条件付き総資源結果

[結果・全gate・shots/energy/総費用](research_c_c1_h4_ca_resources_execution_20261026.md)／[raw19・52round/104軸CSV・SOURCE/calls/STOP](../../artifacts/research_c_c1_h4_ca_resources/2026-10-26/)。固定SOURCE18045333…無変更、119/119 COMPLETE、FAIL/inflight/retry/resume0。直鎖H4・1Å・STO-3G、CDF8/保存元順XDF8、Nα=Nβ2/sector36、全frame S2 s1/s4、δ.1・m0..12、PRなし、own-DF oracle/状態準備除外。4候補の共通CA=.00159 Ha十分条件が数値PASS。CDFの条件付き総RZ削減は同sでs1 11.853846%／s4 11.990819%。保存level B・ideal IID Born・固定serial native modelのpilotで、厳密CA・無条件優位ではない。新fit/DF-PF-合成誤差再計算/build-compile/量子shots抽選0、旧証拠保全・mandatory STOP。以下は実行前準備などの履歴。

## C1：H4 CDF8/XDF8・共通CA/RPE shots/serial資源の準備

[既存再利用・人工/回帰88 PASS・資格の限界](research_c_c1_h4_ca_resources_preparation_20261026.md)／[未承認119-call契約](research_c_c1_h4_ca_resources_contract_unapproved_20261026.json)。新`c1_h4_ca_resources.py`／同名runner・testが、保存52 full512 profile・各DF norm＋guard・共通precision/P0 metricsから既存RPE/Hoeffding/energy/serial費用を接続。直鎖H4・1Å・STO-3G、CDF8/保存順XDF8、Nα=Nβ2/sector36、全frame S2 s1/s4、δ.1・m0..12、PRなし、own-DF oracle/状態準備除外。科学call0、SOURCE24/input12のbyte preflight PASS。新shots/CA/総費用は未評価、厳密enclosure・無条件優位は未資格。旧証拠保全・別明示科学許可までmandatory STOP。

## C1：H4 CDF8/XDF8 fair native/full512限定結果

[結果・全gate・7 metrics・52 profile](research_c_c1_h4_native_fair_execution_20261026.md)／[SOURCE・QPY40・call journal・STOP](../../artifacts/research_c_c1_h4_native_fair/2026-10-26/run01/)。SOURCE862a97d…を無変更実行し305/305 COMPLETE、correctness FAIL/retry/resume0。直鎖H4・1Å・STO-3G、CDF8/保存元順XDF8、Nα=Nβ2/sector36、全frame S2 s1/s4、δ.1・m0..12、PRなし、own-DF oracle/状態準備除外。同H6 Gaussian/P0・既存shared routeで局所18/kernel4/shell3/adjoint8/X-YがPASS、full512数値Frobeniusによる52/52位相条件内。単一h-step RZはCDF12,272対XDF13,944（約11.99%減）。丸めenclosure・共通CA・shots・総資源は未資格/未評価。旧SOURCE/raw/STOPを保全しmandatory STOP。以下の技術準備欄は実行前時点の記録。

## C1：H4 CDF8/XDF8共通precision・full512準備

[技術実装・既存再利用・156 unique人工/回帰PASS・数値資格の限界](research_c_c1_h4_native_fair_preparation_20261026.md)／[未承認305-call科学契約](research_c_c1_h4_native_fair_contract_unapproved_20261026.json)。両armへ同じH6 precision/P0、4 kernel+3 shell、full512 Frobenius、52伝播を固定。実H4新科学0、旧CDF8/XDF10結果とSOURCE不変・mandatory STOP。

## C1：H4保存順XDF8・PF2候補が潜在適格

[限定科学実行・全26セル・SOURCE・資格](research_c_c1_h4_xdf8_pf_execution_20261026.md)。SOURCE328f31007…で93 calls COMPLETE、表現／own-ground/alias／PF26全PASS、s1/s4ともPOTENTIALLY_FEASIBLE。保存DF誤差を再利用、新fit・DF norm再計算・CDF8/XDF10評価・native/合成/CA/shots/費用0。公平な次段階条件は別判断・mandatory STOP。

## C1：H4 XDF8の追加採用・再資格付け準備

[技術報告・既存再利用・人工22＋回帰92 PASS・公平な次段階設計](research_c_c1_h4_xdf8_requalification_preparation_20261026.md)／[未承認93-call契約](research_c_c1_h4_xdf8_pf_contract_unapproved_20261026.json)。保存DF予算PASSを再利用するXDF8専用entryを準備。実H4の表現／own-ground/alias／PF26セル・native／CA／shots／総費用は未実行。旧CDF8/XDF10比較は不変、mandatory STOP。

## C1：H4保存順XDF prefix・DF予算の最初の適格はrank8

[科学実行結果・全9prefix・資格](research_c_c1_h4_xdf_prefix_budget_execution_20261026.md)。SOURCE f7bc730d1…無変更、356 COMPLETE・correctness FAIL0・retry/resume0。rank8 norm＋guard9.5339014551e-5 HaでCA/8以内。旧CDF8/XDF10費用は保持、新XDF8のPF・native・RPE・費用比較は未実施。mandatory STOP。

## C1：案A採択・H4 XDF prefix予算評価の準備

[最小entry・人工29＋旧関連28 PASS・未解決回帰制約](research_c_c1_h4_xdf_prefix_budget_preparation_20261026.md)／[SOURCE・入力・356-call未承認契約](research_c_c1_h4_xdf_prefix_budget_contract_unapproved_20261026.json)。保存順1..9を全件測定し、保存rank10 anchorと比較する計画。実H4のdecode／新norm／rank判定0。科学実行は別途明示許可後のみ、mandatory STOP。

## C1：H2／H4／H6比較条件・baselineの技術監査

[統合表・H4 XDF10選定来歴・native精度差分](research_c_c1_h2_h4_h6_comparison_audit_20261026.md)／[最小実験A・Bの独立設計](research_c_c1_minimal_experiment_design_20261026.md)。保存値の引用と静的SOURCE監査、人工29 PASSのみ。H4低prefixのCA/8適格性は未評価、H6 rank13の開始予算gateの違いを明記。案Aを技術推奨するが未採択・新科学call0・mandatory STOP。

## C1：H6共通CA・shots・条件付き総資源pilot完了

[4候補数値CA PASS・総資源表](research_c_c1_h6_ca_resources_execution_20261026.md)。SOURCE55b3d868…無変更・119 calls・52round PASS。同sの全N一致、総RZはXDFが小さい。厳密C/CA/device/状態準備込み/4候補同時/無条件保証には昇格しない。旧証拠保全・mandatory STOP。

## C1：H6共通CA・RPE shots・条件付き総資源の技術準備

[接続方式・102人工/回帰PASS・資格の限界](research_c_c1_h6_ca_resources_preparation_20261026.md)／[未承認119-call契約](research_c_c1_h6_ca_resources_contract_unapproved_20261026.json)。保存H6の資格B・DF/PF・native metricsを既存評価器へ接続。候補単位failureとserial費用定義を保持し、実H6新評価0・mandatory STOP。

## C1：H6全基底合成上界・限定実行結果

[結果・8経路上界・全104位相条件PASS](research_c_c1_h6_synthesis_bound_execution_20261026.md)。SOURCE e36e056…無変更、G4×800列、3335 calls、FAIL0。資格B／厳密Cを区別、CA/shots/cost0、旧結果保全・mandatory STOP。

## C1：H6合成誤差上界の技術準備

[方式比較・数学的条件・人工/回帰134 PASS](research_c_c1_h6_synthesis_bound_preparation_20261025.md)／[未承認6535-call契約](research_c_c1_h6_synthesis_bound_contract_unapproved_20261025.json)。全sector基底streaming数値上界を推奨。厳密C・CA・shots・総費用は未資格、実H6追加科学0・mandatory STOP。

[C1 H6 native修正後・298-call全PASS](research_c_c1_h6_native_stable_phase_execution_20261025.md)：SOURCE3d7b56b448…無変更、局所30/kernel8/shell3/adjoint8/X-Y8 PASS、QPY52・104未資格profile保存。CDF14のkernel RZはXDF14より約1.98%/1.96%多く、literal/shared費用差0。CA/総費用は未資格、旧FAIL保全・累積627 units・mandatory STOP。

[C1 H6 P0 phase安定化](research_c_c1_h6_native_stable_phase_repair_20261025.md)：長い人工回路で旧4.35e-10→修正5.36e-13、gate本体は完全一致。97 tests PASS、[fresh298-call条件](research_c_c1_h6_native_stable_phase_contract_20261025.json)を固定。

[C1 H6 native Gaussian精度修正](research_c_c1_h6_native_precision_repair_20261025.md)：既存Nature JWの数値処理とactual inverse/P0競合を修正。人工185 PASS、旧162-call STOPを保持。同一8経路・新SOURCE44・fresh298-call再実行条件を固定。

[C1 H6 native限定実行・部分STOP](research_c_c1_h6_native_execution_20261025.md)：SOURCE bf431765…無変更、162/298 units（154 COMPLETE／8 FAIL）。局所30／shell3 PASS、最初のraw/reference差1.0400731857e-10>固定1e-10。kernel8 bundle FAIL、compiled correctness・adjoint/X-Y・kernel費用未資格／未到達。QPY52・旧証拠保全、retry/resume0、mandatory STOP。

[C1 保存H6 rank14・S2 PF結果](research_c_c1_h6_saved_pf_execution_20261025.md)：SOURCE87ea924ae…、72/72 calls、一般K/ground/alias全PASS、CDF14/XDF14のs1/s4全52roundが登録PF条件PASS。4候補は潜在適格、合成誤差・CA・総資源は未資格。新fit/native/shots/cost0、旧証拠保全・mandatory STOP。

[C1 H6 rank14修正codec・1 fit結果](research_c_c1_h6_corrected_fit_execution_20261025.md)：固定SOURCE33432e786…、20/20 calls、optimizer成功とDF予算PASS。初期tensor F減少／sector誤差4.3116%増を区別し、旧CDF14比sector93.9132%減を保存。生x/jac・STOPを公開し、PF/native/shots/costは未評価。mandatory STOP。

[C1 Stage2 native科学結果](research_c_c1_stage2_native_execution_20261025.md)：固定SOURCEで60主検査・内包unitarity52件が全PASS、call243/243、fit／optimizer／retry／resume0。費用はCDFが小さいがCDF/XDFの物理精度差を保持し、同精度総資源とはしない。原本 `artifacts/research_c_c1_stage2_science/2026-10-25/`、Stage3へ進まずmandatory STOP。下記の準備時点の未承認表示は履歴。

[C1 Stage2 native接続準備](research_c_c1_stage2_native_preparation_20261025.md)：CDF rank8の限定対象採択、旧独立CAR／Gaussian／controlled／X/Y／compiler資産を再利用。人工80件＋29 subtests、52 pins・byte preflight PASS。[未承認科学契約](research_c_c1_stage2_native_contract_unapproved_20261025.json)は局所全PASS後にfull/adjoint/X/Y/compile、上限243 unitsの推奨案。科学条件未採択・実H4 decode／実回路／科学call0、証拠 `artifacts/research_c_c1_stage2_technical/2026-10-25/`、mandatory STOP。

[C1 保存fit評価専用科学実行](research_c_c1_stage1_eval_only_execution_20261025.md)：37 gate PASS、full256 frame0再検証・sector36 adapter同値性PASS。保存CDF rank8／XDF rank10の物理誤差は別測定、新fit／optimizer0、科学call141/141、retry/resume0。原本 `artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/`、旧STOP保全、Stage2は別判断・mandatory STOP。下記の未承認・未再検証表示は当時の履歴。

[C1 保存fit評価専用経路の技術準備](research_c_c1_stage1_eval_only_preparation_20261025.md)：新規fit0、旧fit来歴と新評価SOURCEの独立検証、既存CAR/K/XDFを再利用。人工82件＋29 subtests PASS、固定byte/metadata preflight PASS。[未承認科学契約](research_c_c1_stage1_eval_only_contract_unapproved_20261025.json)は最大141、fresh output、retry/resume0、mandatory STOP。証拠 `artifacts/research_c_c1_stage1_eval_only_technical/2026-10-25/`、H4数値decode・評価なし、旧FAIL未再検証。

[C1 raw CAR frame縮約修正](research_c_c1_frame_repair_20261025.md)：旧rawの起動／読込順に依存する人工並列不一致を再現し、explicit matmulへ変更。人工63件＋29 subtests・別プロセス64 frame比較PASS。[未承認引継ぎ](research_c_c1_frame_repair_handoff_unapproved_20261025.json)、証拠 `artifacts/research_c_c1_frame_repair/2026-10-25/`。H4追加解析・fit0、科学的資格は未完了、旧STOP保全・mandatory STOP。

[C1 Stage 1互換性修正版の科学実行](research_c_c1_stage1_compat_execution_20261025.md)：新しい1 fitはrank8で収束、full256 frame0同値性FAILでSTOP。κ/J/U・baseはPASS、CDF/XDFの物理sector誤差は未測定。科学call107/147、retry/resume0、証拠 `artifacts/research_c_c1_stage1_compat_science/2026-10-25/`。旧SOURCE／run／STOPを保全し、追加科学処理・Stage2/3へ進まない。

[C1 Stage 1：ffsim／SciPy互換性修正](research_c_c1_stage1_compatibility_20261025.md)：既存2D logmを順に呼ぶ専用bridge、venv起動修正、人工59件＋29 subtests PASS。[新未承認契約](research_c_c1_stage1_compatibility_contract_unapproved_20261025.json)とSOURCE／lockは `artifacts/research_c_c1_stage1_compatibility/2026-10-25/`。実分子科学call0、旧結果／STOP保全、別の1 fit明示承認待ち・mandatory STOP。

このディレクトリには、DF部分ランダム化・有限RTE・RPE compiled cost研究の目的、背景、未解決点、解析手順および数値評価計画をまとめる。

実装、runner、テスト、成果物を含むリポジトリ全体の区分は
[`../../PROJECT_MAP.md`](../../PROJECT_MAP.md)を参照する。

初めてこの研究を確認する場合や、Codexを使って発表・共有資料を作る場合は、まず
[研究概要・現状](研究概要・現状.md)を読む。この一冊で現在の研究段階、採用済みの
方針、主要な検証結果、未決定事項および証拠の読み順を確認できる。

## 文書の役割

研究Cの[C4終了・成果引継ぎ](research_c_c4_closeout_handoff_20261024.md)はユーザー採択済みの一時終了と限定結論・B2等の資産・再開条件の正本。[C1圧縮DF接口調査](research_c_compressed_df_interface_survey_20261024.md)は外部実装・一般対角fragment・誤差/位相・既存SOURCEの再利用と未承認の次案を整理する。C1方式の採択・新実装・科学実行は含まない。

Gaussian singletonの[改善候補設計](pr_gaussian_singleton_synthesis_design_20261024.md)は、support1／support2を区別した選択部分空間Givens案と、既存helperの再利用・同値性条件・未承認の48-wrapper対照案。静的調査のみで、削減量未測定・候補未実装。

限定F2の[保存case寄与分析](pr_algorithm_improvement_phase2d_f2_attribution_20261024.md)は、8分類表のP・条件付き費用・寄与とexact全体照合をまとめる。SP／PS差の記述分解、限界、次の改善候補を明示し、新build／compileは行っていない。

限定F2の[回路資源差の原因調査](pr_algorithm_improvement_phase2d_f2_resource_causes_20261024.md)は、固定SOURCEと保存caseの静的照合による技術メモ。Gaussian往復・共有条件・K0の成分選択と、未測定の寄与を区別する。新しい数値集計・build／compileは行っていない。

| 種類 | 正本とする内容 |
|---|---|
| [研究概要・現状](研究概要・現状.md) | 現在地と資料作成用の統合要約 |
| 下記の主資料4本 | 研究目的、理論的位置付け、解析方法、評価計画 |
| `docs/*.md`の検証資料 | 個別検証の方法、条件、数値結果、限界 |
| `VALIDATION_STATUS.md`とmanifest | 外部再現性、証拠status、利用禁止結果 |
| `artifacts/` | machine-readableな数値結果とprovenance |
| 研究ノート | 判断の経緯。現行仕様の正本ではない |

主資料4本だけで研究設計を追えるようにし、補足Q&Aは式の直感、記号の違いおよび
具体例を確認するために用いる。

## 推奨する閲覧順序

### 主資料

| 順序 | 資料 | 説明 |
|---:|---|---|
| 1 | [研究目的・研究課題](研究目的・研究課題.md) | 研究目的、現段階の範囲、最終目的関数およびResearch Questionsを示す。 |
| 2 | [先行研究と未解決点](先行研究と未解決点.md) | PR論文の理論・resource modelと、本研究で扱う有限・compiled-cost上の未解決点を整理する。 |
| 3 | [研究方法・解析手順](研究方法・解析手順.md) | 記号、入力、探索変数、誤差・shot・期待compiled costの依存関係と最適化手順を定める。 |
| 4 | [数値実験・評価計画](数値実験・評価計画.md) | 現行実装で評価できる範囲、固定条件、実験手順、比較方法および判定基準を定める。 |

### 補足資料

| 順序 | 資料 | 説明 |
|---:|---|---|
| 1 | [研究目的・研究課題 補足Q&A](研究目的・研究課題_補足QA.md) | normalization、attenuation、位相誤差換算の直感を補う。 |
| 2 | [先行研究と未解決点 補足Q&A](先行研究と未解決点_補足QA.md) | PR論文の各量の意味と有限RTEへの対応を具体化する。 |
| 3 | [研究方法・解析手順 補足Q&A](研究方法・解析手順_補足QA.md) | 外側・内側最適化、信号半径および数値例を詳しく説明する。 |

## 研究ノート

日付ごとの実装方針、判断理由、検証結果および次の課題は、
[研究ノート](研究ノート/README.md)に時系列で記録する。研究ノートは変更の
経緯を残すための資料であり、現行仕様は上記の主資料、検証可能性と保証statusは
`VALIDATION_STATUS.md`およびmachine-readable artifactを正本とする。

## 実装・検証資料

現行実装の保証範囲、検証状況およびAPI規約は、研究資料の閲覧順序とは分けて次を参照する。

- [検証状況](../../VALIDATION_STATUS.md)
- [有限RTE規約](../rte_conventions.md)
- [finite-RTE信号近似の小規模検証](../finite_rte_signal_validation.md)
- [RPE短roundの信号・shot・回路cost接続検証](../rpe_round_cost_connection_validation.md)
- [RPE短roundの仮想Hadamard測定・失敗確率検証](../rpe_hadamard_failure_validation.md)
- [$q=8$ Hadamard 1 shot cost proxy・resource接続検証](../rpe_hadamard_proxy_resource_validation.md)
- [RPE位相誤差・失敗確率配分の感度検証](../rpe_allocation_sensitivity_validation.md)
- [RPE限定4段の集計・失敗確率検証](../rpe_four_round_accounting_validation.md)
- [RPE 4段の物理信号・分枝復元検証](../rpe_four_round_phase_validation.md)
- [目標精度からのRPE round範囲・固定設定診断](../rpe_target_round_horizon_validation.md)
- [RPEのdelta候補とround別有限RTE schedule検証](../rpe_delta_round_schedule_validation.md)
- [delta scheduleの中央RTE compiled-cost proxy検証](../rpe_delta_compiled_cost_validation.md)
- [ランダム回路compiled-cost加法モデルのpilot検証](../random_circuit_cost_validation.md)
- [RTE境界補正cost modelのpilot検証](../rte_boundary_cost_validation.md)
- [RTE境界補正のfragment層別・高統計検証](../rte_boundary_pair_validation.md)
- [階層compiled-cost modelの拡張holdout検証](../hierarchical_cost_validation.md)
- [RTE connected-cluster運用cost推定の独立holdout検証](../rte_connected_cluster_cost_validation.md)
- [ランダムRTE回路compiled-cost近似検証の統合結果](../rte_compiled_cost_validation_summary.md)
- [PF誤差surrogate・CPU摂動・QPE分枝のholdout検証](../pf_delta_validation.md)
- [H-chain系サイズ・実行可能delta窓におけるPF係数検証](../pf_c_system_size_validation.md)
- [RPE resource accounting](../rpe_resource_accounting.md)
- [RTE一次資料の版管理](../rte_source_versions.md)

## C1 H4強baseline：技術準備（2026-10-26）

[技術報告・合法なbasis自由度・位相補正・人工101 PASS](research_c_c1_h4_strong_baseline_preparation_20261026.md)／[未承認227-call契約案](research_c_c1_h4_strong_baseline_science_contract_draft_20261026.json)。保存CDF8/XDF8 baseline、XDF spin基底単独、両armのrelative Gaussian単独をs1で切り分ける。B3・s4・13round総資源は未登録、実H4新科学call0。別途の明示的な科学実行許可までmandatory STOP。

## 2026-10-26：H4強baseline s1科学実行完了

[227-call結果・full512検証・5比較](research_c_c1_h4_strong_baseline_execution_20261026.md)。9 local／3 kernel／6 adjoint／9 complex X/Y gate PASS。B1 XDF RZ11821、B2 CDF11062/XDF11307、保存B0は再buildなし。単一kernelの数値pilotで、B3・s4・CA/shots/総費用は未評価。旧結果・SOURCE・STOPを保全し、公開照合後mandatory STOP。

## C1 H4 B3：技術準備（2026-10-26）

[技術報告](research_c_c1_h4_b3_preparation_20261026.md)／[未承認65-call案](research_c_c1_h4_b3_science_contract_draft_20261026.json)。保存e0c89f61結果・元reference・shell・own-groundを再利用し、B3 XDF8／s1/h.1の一件だけを新規検証する。新21＋旧101＝122 unique tests PASS、SOURCE35／直接input19・旧binding byte preflight PASS。実H4新評価0、旧SOURCE/結果/STOPと研究状態を保全、別明示許可までmandatory STOP。

## 2026-10-26：H4 B3 s1・65-call限定実行完了

[全結果・full512 gate・6指標・3比較](research_c_c1_h4_b3_execution_20261026.md)。固定SOURCE b40fce5c…無変更、65/65 COMPLETE、correctness FAIL/retry/resume0。H4直鎖1Å/STO-3G、元順XDF8/保存CDF8、Nα=Nβ2、全frame S2 s1/h.1、PRなし、own-ground/状態準備除外、共通precision/P0。新B3 spin＋relative一件のみbuild/compileし、raw/P0/actual adjoint/X-Yは全512入力・出力で数値PASS。RZ10885/CX6892/totaldepth11973は保存B2 CDF11062/7008/12127より小さく、他3指標も小さい。単一kernel比較で、同CA総順位・13round・shots・費用・厳密enclosureは未評価／未資格。旧raw・SOURCE・STOP保全、mandatory STOP。上の準備・未実行表記は履歴。

## 2026-10-26：H4 B2 CDF／B3 XDFのCA・RPE・総資源の技術準備

[再利用・接続・人工76 PASS・資格の限界](research_c_c1_h4_strong_baseline_ca_resources_preparation_20261026.md)／[未承認86-call契約](research_c_c1_h4_strong_baseline_ca_resources_contract_draft_20261026.json)。新module `c1_h4_strong_ca_resources.py`／同名runner・testsが保存full512 segment・norm guard・shell・元PF13を既存propagate/RPE/energy/serial集計へ接続する。H4直鎖1Å/STO-3G、CDF8/XDF8、Nα=Nβ2、全frame S2 s1のみ・δ.1/m0..12、PRなし、own-ground/状態準備除外、共通precision/P0/CA.00159。回路／Operator再生成不要、実科学call0。旧SOURCE/raw/STOP・旧約12%総資源とB3単一kernel結果を保全し、新総順位・厳密enclosureは未評価／未資格。公開照合後mandatory STOP。

## 2026-10-26：H4強baseline CA・実行前binding修正

[原因・人工78 PASS・修正契約v2・実科学call0](research_c_c1_h4_strong_ca_binding_repair_20261026.md)。旧86-call契約の保存contract参照が、同一bytesのdocs pathとartifact pathの文字列差でSTOPするため、canonical参照だけを修正した。科学SOURCE31件・数値条件・call上限86・旧契約/raw/STOP不変。run01未作成・未消費、科学実行は未開始。修正契約での別明示許可までmandatory STOP。

## 2026-10-26：H4強baselineの共通CA・serial総資源比較完了

[全結果・26round・shots・総費用・資格の限界](research_c_c1_h4_strong_baseline_ca_resources_execution_20261026.md)。固定SOURCE1aeedd961…／binding-v2無変更、86/86 COMPLETE、correctness FAIL/retry/resume0。H4直鎖1Å/STO-3G、B2 CDF8/B3 XDF8、Nα=Nβ2、全frame S2 s1のみ・δ.1/m0..12、PRなし、own-ground/ideal IID Born/状態準備除外、共通precision/P0/CA.00159。両候補のCA数値十分条件PASS、shots13606/13600、条件付き総RZ102056444569/100267421030でXDFが1.75297%少ない。level B数値pilotの限定比較で、厳密enclosure・無条件／一般の優位は未資格。旧結果保全、公開照合後mandatory STOP。上の準備・未実行表示は履歴。
