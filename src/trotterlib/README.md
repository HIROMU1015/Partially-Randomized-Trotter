# trotterlib モジュール索引

## 研究C：第三γ0 DF・141-callでnative metadata技術STOP（2026-10-29）

[部分結果・全13round・STOP原因](../../docs/research/research_c_df_basis_third_execution_20261029.md)。保存H4/1Å/STO-3G、γ0 CDF8/LD2、partial S2 δ=h=.1/s1/r128/K0、own-ground/状態準備除外。SOURCE14270239a無変更で141 attempts=140 COMPLETE+1 technical FAIL。表現/ground/alias・13/13 PF-RTE十分条件PASS、ideal仮定shots21754。Native full_law_gateへ渡すmetadataのcomponents欠損でSTOPし、sampling/build/compile/DF holdout0。第三DF価格/予測誤差/総費用はnull、旧C不適格不変。SOURCE112/input93/旧保護1667件照合、新fit/retry/resume0、旧原本保全。公開照合後mandatory STOP。以下の未実行表記は当時の履歴。

## 研究C：basis-switch-aware設計・保存γ0第三DFの準備（2026-10-29）

[数式・再利用・人工検証・未承認493-call案](../../docs/research/research_c_df_basis_switch_optimization_design_20261029.md)。固定exact-key partitionのswitch項はJのmass比に依存し、自由Uの幾何を直接最適化しない。Exact/smooth零tail・ε bias・非凸性・gaugeを分離し、η/新目的は未採択。保存H4/1Å/STO-3G、γ0 CDF8/LD2を第三候補としてown h=.1/s1/r128/K0の13round、条件付きRPE、GO時native32、DF holdoutへ接続。旧2cohortの共通interceptを第三価格前に固定し、Cは復活させない。新21＋回帰110＝131 PASS、科学call0、SOURCE112/input93/旧原本1667件照合、旧103 SOURCE/STOP不変。別科学許可までmandatory STOP。

新module `c_df_basis_switch.py`（純粋PoC）、`c_df_basis_transfer.py`（DF holdout）、`c_df_basis_third.py`／`c_df_basis_third_native.py`（候補接続）。入口 `scripts/run_research_c_df_basis_third.py` はpassive既定。人工testは `tests/management/test_research_c_df_basis_switch*.py`／`test_research_c_df_basis_third.py`。

## 研究C：basis-key修正後148-call完了・複合surrogate未資格（2026-10-29）

[結果・モデル比較・旧STOP保全](../../docs/research/research_c_df_basis_surrogate_repair_execution_20261029.md)。保存直鎖H4・1Å/STO-3G、CDF8/LD2、partial S2 δ=h=.1/s1/r128/K0、M4 Proposed/tail-P4各32件。既存spin_liftによる厳密key修正、新7＋回帰43＝50 PASS、SOURCE030128649…を固定しfresh run02で148/148 COMPLETE、correctness FAIL0。RZ MAEはM4でA221.41<C227.80、P4でB261.54<C273.58のため両cohortともSURROGATE_NOT_QUALIFIED。全192 fold full rank・負予測0。静的DFは2個、因果/新DF汎化/η採択は未資格。新fit/sampling/build/compile0、旧SOURCE99/run01/保護1570件不変。技術relaunch1、run内retry/resume0、公開照合後mandatory STOP。以下の未実施/旧STOP表記は履歴。

実行入口 `scripts/run_research_c_df_basis_surrogate_v2.py`、接続 `src/trottertracks/c_df_basis_reanalysis_v2.py`、key修正 `c_df_basis_surrogate_v2.py`、人工test `tests/management/test_research_c_df_basis_surrogate_v2.py`。消費済みrun02は再起動しない。

## 研究C：basis-aware saved64・3-callで技術STOP（2026-10-29）

[部分証拠・原因候補・残る修正](../../docs/research/research_c_df_basis_surrogate_execution_20261029.md)。保存直鎖H4・1Å/STO-3G、CDF8/LD2、PR s1/r128/K0の2候補各32件を対象に固定SOURCEb1985ff01で開始。3 attempts=2 COMPLETE+1 FAIL、M4_Proposed/static_featuresのEXACT_SAVED_BASIS_KEYでSTOP。Trajectory/QPY/model未実施、性能未判定。既存spin_liftと新np.kron経路のsigned-zero repr差を人工2×2だけで再現。SOURCE/契約変更・retry/resume0、旧1458件保全、公開照合後mandatory STOP。

## 研究C：basis-aware native surrogateの技術設計（2026-10-29）

[3案・実装対応・2cohort schema・未承認148-call案](../../docs/research/research_c_df_basis_native_cost_surrogate_design_20261029.md)。切替/overlap/複合run-atomを比較し、複合特徴ベクトルを推奨。新12＋回帰31＝43人工PASS。保存H4の新特徴量・相関・NPZ/QPY数値decode・fit/sampling/build/compileは0。η/scale/新目的は未採択。2静的DFを64標本扱いせず、次は保存各32件の独立cohort内記述検証だけを別許可で実行。旧原本保全、公開照合後mandatory STOP。

## 研究C：tail-P4 PRの449-call完了・総資源改善なし（2026-10-28）

[PF/RTE・shots・損益分岐・native・総資源](../../docs/research/research_c_df_structural_pr_performance_20261028.md)。保存直鎖H4・1Å/STO-3G、tail-P4 CDF8/LD2、partial S2 δ.1/s1/r128/K0/m0..12、own-DF oracle/状態準備除外。SOURCE4ac0e5f96無変更、449/449 COMPLETE、correctness FAIL/retry/resume0。全13round適格、主仮定shots16,568、fixed32/full512/adjoint/X/Y PASS。Kernel平均RZ15,047.46875、M4-r128比1.068983。条件付き総RZ275,422,425,481.5/旧234,117,788,135=1.176427。疎化による価格/総資源改善は今回の点比較では観測なし。全law/actual CA/母平均優位は未資格。旧1054件保全、公開照合後mandatory STOP。

## 研究C：tail-P4のPR・条件付きnative32経路（2026-10-28）

[再利用・40人工PASS・449-call段階付き契約](../../docs/research/research_c_df_structural_pr_preparation_20261028.md)。候補固有role、保存J/U固定、new13round/ground/law、登録GO時だけnative32。科学開始前SOURCE固定、旧gate/source不変、未知全law誤差はnull。

## 研究C：tail J固定support CDFの1 fit・DF予算PASS（2026-10-28）

[結果・SOURCE・call・STOP](../../docs/research/research_c_df_structural_support_execution_20261028.md)。保存直鎖H4・1Å/STO-3G、CDF8/LD2、head-full/tail-P4、γ.1。SOURCE59e186ad無変更、21/21 COMPLETE、1 fit、retry/resume0、correctness FAIL0。終了sector誤差7.45156e-9Ha＋guard3.26388e-9HaでCA/8 PASS。Tail係数位置144だがλ1.51617Haは旧PR-aware1.44243Haより大きく、性能GOではない。PF/RTE/shots/native/総費用0、旧1006件保全、公開照合後mandatory STOP。

## 研究C：DF構造改良の技術提案・科学実行未承認（2026-10-28）

[横断調査・3案・再利用地図・最小契約](../../docs/research/research_c_df_structural_design_20261028.md)。保存H4・1Å/STO-3G、CDF8/LD2を維持し、既存γ/r探索を拡張せずtail Jの固定support制約を最初の検証案とした。既知support制約CDFの限定拡張で独立新規性/資源優位は未認定。旧loss/AD/codec/SciPyを再利用するcompact adapter、1-fit限定runner、人工60 tests＋29 subtests PASS。未承認21-callはDF評価だけで、実H4decode/fit/PF-RTE/shots/native0。後続は新J/U固有の別契約が必要。旧SOURCE/raw/STOP保全、remote照合後mandatory STOP。

## 研究C M4-Native128：固定32校正・332-call完了（2026-10-28）

[full512・全36判定・条件付き資源](../../docs/research/research_c_pr_aware_df_M4_native128_execution_20261028.md)。保存直鎖H4・1Å/STO-3G、Proposed CDF8/LD2、partial S2 δ.1/s1/r128/K0/m0..12、own-DF oracle/状態準備除外。旧87SOURCE無変更、説明128化の独立revision＋metadata test、人工回帰66 PASS。332/332 COMPLETE、32 full512/actual adjoint/X/Y PASS、retry/resume0。P0平均RZ 14076.437500、ideal仮定総RZ 234117788135.000、M3r64比0.793138/同JU決定論比2.299196。M4保存shots16056/W/S/36閾値を再利用、新PF/RTE/shot/CA/fit0。有限cohort条件付き点比較で、母平均/actual CA/全law/support未資格。旧642件保全、公開照合後mandatory STOP。前段のnative未実行表記は当時の履歴。

## 研究C M4：r128・保存PF/shots/損益分岐98-call完了（2026-10-28）

[結果・全13round・6指標閾値・次段階案](../../docs/research/research_c_pr_aware_df_M4_s1_r128_feasibility_breakeven_20261028.md)。新`c_pr_aware_df_m4.py`/専用runnerはM3PF13を再利用して98/98 COMPLETE、FAIL/retry/resume0。保存Proposed CDF8/LD2/δ.1/s1/r128/K0・own-DF oracle/状態準備除外。全13適格、主仮定shots16056。未知kernel価格のRZ損益分岐はM3比17747.772397、同J/U決定論比6122.327980。実r128価格/総費用はnull、実sampling/build/compile0。別native adapter/CLIは未承認332-call/32sample案、人工5のみ。SOURCE/旧原本保全、公開照合後mandatory STOP。

## 研究C M3：JSON修正後のs1/r64・471-call完了（2026-10-28）

[結果・全13round・32 native校正・条件付き資源](../../docs/research/research_c_pr_aware_df_M3_s1_r64_repair_execution_20261028.md)。修正SOURCE 5354edecc…を固定しfresh run02で471/471 COMPLETE、correctness FAIL0。全13適格・32 full512/adjoint/X/Y PASS。保存Proposed CDF8/LD2/δ.1/s1/r64/K0・own-DF oracle/状態準備除外、ideal仮定のshots 20734、総RZ 295179018953、同J/U決定論比 2.89885907。実装CA/全law/母平均優位は未資格。旧run01の19-call schemaSTOPを保持、技術relaunch1/run内retry-resume0。公開照合後mandatory STOP。

## 研究C M3：s1/r64・A途中の技術STOP（2026-10-28）

[結果・原因・19-call部分証拠](../../docs/research/research_c_pr_aware_df_M3_s1_r64_execution_20261028.md)。SOURCE9752f28cf…を変更せず一度実行し19 attempts/18 COMPLETE/1 schema FAIL。Atom catalogのtupleと保存JSONのlistを直接比較した不備でSTOP。PF/RTE round0、B/C未実行、科学的不適格は未判定。新`c_pr_aware_df_m3.py`／`run_research_c_pr_aware_df_M3.py`／人工test、92 PASSとJSON境界の限界を記録。Retry/resume0、旧原本保全、mandatory STOP。

## 関連track：M2同J/U決定論adapter（2026-10-28）

[結果・再利用・68人工/回帰・253-call契約](../../docs/research/research_c_pr_aware_df_M2_same_JU_deterministic_execution_20261028.md)。新`c_pr_aware_df_m2.py`／`run_research_c_pr_aware_df_M2.py`／同名人工test。両full512/CA数値条件PASS、旧PR仮定に対し今回決定論s1が低費用。SOURCE無変更、retry/resume0、消費済みrun01を再起動せずmandatory STOP。

## 関連track：M1保存値adapterの限定実行完了（2026-10-28）

[全結果・20-call・未資格事項](../../docs/research/research_c_pr_aware_df_mechanism_execution_20261028.md)。20 COMPLETE、correctness FAIL/retry/resume0、科学SOURCE変更0。旧2仮定でshots側項が価格側より大きい記述的分解。消費済みM1-run01を再起動せず、因果/CA/強baseline順位を認定しない。公開照合後mandatory STOP。

## 関連track：保存値M1の記述的機構adapter（2026-10-28）

[技術報告・SOURCE・未承認契約](../../docs/research/research_c_pr_aware_df_performance_mechanism_readiness_20261028.md)。`c_pr_aware_df_mechanism.py`／`run_research_c_pr_aware_df_mechanism.py`、passive既定・科学許可後だけ`--execute`。次の一実験には保存shots×保存whole-kernel価格の4corner・記述的寄与分解M1を推奨。新adapter/CLI/testの3ファイル、人工新30＋回帰23＝53 PASS、SOURCE7/input9・未承認20-call契約を固定した。因果効果/actual CA/保証shots/母平均優位への昇格なし。新科学call0、M1-run01未作成。公開照合後mandatory STOP、別明示許可まで再解析・新fit・PF/RTE・build/compile・sampling・移送・詳細レビューを開始しない。

## 関連track：Proposed F3-K1科学実行完了（2026-10-28）

[結果・証拠・資格範囲](../../docs/research/research_c_pr_aware_df_F3_K1_execution_20261028.md)。既存CAR/Arb/bridge/executorを無変更で使用し、7/7 COMPLETE、全512基底のraw/K1≤3.42202E-14、compiled/K1≤3.24349E-12（上向き表示）。新fit/build-compile/sampling0。固定一trajectoryのみ、全law/CA/資源未資格・mandatory STOP。

## 関連track：Proposed F3-K1 binding（2026-10-28）

`../trottertracks/c_pr_aware_df_fixed_f3_k1.py` は既存K1 CAR/Arb/bridge/executorを再利用するProposed専用adapter。[F1 reuse条件・人工193 tests＋29 subtests・7-call案](../../docs/research/research_c_pr_aware_df_F3_K1_readiness_20261027.md)。新実H4科学call0、F3未実行・mandatory STOP。

## 関連track：F2-K1科学実行完了（2026-10-27）

[固定Baseline・全512基底・validated CAR結果](../../docs/research/research_c_pr_aware_df_F2_K1_execution_20261027.md)。SOURCE無変更、7 COMPLETE/correctness FAIL0/retry-resume0。Exact κH/polar Qへのraw/K1≤2.67040e-14、compiled/K1≤3.49618e-12（上向き表示）。旧SOURCE/raw/STOP不変。全law/CA/保証shots/総資源は未資格、公開照合後mandatory STOP。

## 関連track：F2-K1技術準備（2026-10-27）

`../trottertracks/c_pr_aware_df_fixed_k1.py`／`fixed_k1_execution.py` は[exact κH・polar Q・2 bridgeと独立CARの新target adapter](../../docs/research/research_c_pr_aware_df_F2_K1_target_readiness_20261027.md)。既存CAR/Arb/全基底backendとCallsを再利用。人工153 tests＋29 subtests PASS、実H4科学call0、旧F2厳密gate不変、未承認7-call契約・mandatory STOP。

## 関連track：F2 κ限定診断完了（2026-10-27）

[実行結果](../../docs/research/research_c_pr_aware_df_F2_kappa_diagnostic_execution_20261027.md)：SOURCE無変更で6/6 COMPLETE、correctness FAIL/retry/resume0。κはStage 1数値allowance内だがexact Hermitianではなく、入力と演算の両方に寄与。CAR target/旧F2 STOPを変更せずmandatory STOP。

## 関連track：F2 κ限定診断（2026-10-27）

`../trottertracks/c_pr_aware_df_kappa_diagnostic.py` は既存corrected_kappa・allowance・Callsを再利用するopt-in診断。保存値のexact有理数縮約で入力由来／演算由来を区別し、入力・CAR targetは変更しない。[SOURCE・契約・人工検証](../../docs/research/research_c_pr_aware_df_F2_kappa_hermiticity_diagnostic_readiness_20261027.md)。新規実H4科学call0、別科学許可までmandatory STOP。

## 2026-10-27：固定F2 polar・κの厳密Hermiticity条件でSTOP

[結果・partial evidence](../../docs/research/research_c_pr_aware_df_fixed_F2_polar_execution_20261027.md)。保存直鎖H4・1Å/STO-3G、通常number CDF8 Baseline baseline_00、元順LD2、partial S2 δ.1/s4/h1/40/r16/K0、sector36/full512、own-DF oracle/状態準備除外。Right-polar CAR targetを今回限定採択し、SOURCE45ae98943492e921c22a0f527d5587cd1e06462f／新契約を固定して実行。4/最大7 attempts＝3 COMPLETE＋1 FAIL、残3 NOT_RUN。CAR_reference入口の元g補正κがEXACT_HERMITIAN_INPUT_REQUIREDでSTOP。Raw/CAR・bridge・triangleは未評価/nullで、raw回路の測定correctness FAILではない。追加診断・κ変更・retry/resume・F3なし。旧B1/F1/C1の資格とraw/STOP不変、公開照合後mandatory STOP。

## 2026-10-27：固定CAR reference・Proposed検証の技術準備

`trottertracks.c_pr_aware_df_fixed_car` は既存CAR/atom/event/head関数へopt-in Arb演算を接続する。`fixed_car_execution` は既存full-basis verifierのProposed拡張とCAR adapter。旧B1 runner不変。[技術報告](../../docs/research/research_c_pr_aware_df_fixed_car_reference_readiness_20261027.md)。F1未承認7 calls、F2 target未採択・実行不可、実H4科学call0。

## 2026-10-27：固定回路のArb全基底事後検証

`trottertracks.c_pr_aware_df_fixed_certificate`は既存Arb gate意味論を再利用し、固定gate列の全計算基底columnをrecentered local actionで囲う。`c_pr_aware_df_guarantee_strategy`はuniformとcaptured・統計scopeを区別、`guarantee_execution`は未承認の保存QPY19-call案。[技術報告](../../docs/research/research_c_pr_aware_df_c2a_guarantee_strategy_readiness_20261027.md)／[契約](../../docs/research/research_c_pr_aware_df_c2a_guarantee_contract_draft_20261027.json)。新37＋旧193 PASS。Raw physical/reference・全law・CA/総資源は未資格、実H4科学call0。

## 2026-10-27：ZSXX全到達domain・条件付きwhole-P0

`trottertracks.c_pr_aware_df_zsxx_uniform`は旧Arb192 evaluatorとmonotone_budgetを再利用し、source domain、run長依存式、partial translation規則、true-period phase項、条件付きwhole-P0 composerを追加する。Captured/ideal boxを実Rust imageとして拒否する。[技術報告](../../docs/research/research_c_pr_aware_df_zsxx_uniform_domain_readiness_20261027.md)／[実行不可契約](../../docs/research/research_c_pr_aware_df_zsxx_uniform_contract_draft_20261027.json)。新規32＋旧161 PASS、実H4科学call0、旧SOURCE保全。

## 2026-10-27：C2-A・ZSXX captured-pair validated enclosure

Opt-in `trottertracks.c_pr_aware_df_zsxx_enclosure`はactual Rust入力・output bitsをArb192で独立検証し、phase-sensitive spectral距離のoutward Frobenius上界を返す。人工fixture／passive契約helperも追加。[技術報告](../../docs/research/research_c_pr_aware_df_c2a_zsxx_enclosure_readiness_20261027.md)／[実行不可契約](../../docs/research/research_c_pr_aware_df_c2a_zsxx_contract_draft_20261027.json)。固定pairとheld-V reference領域のみで、全law・初期translationの一様上界は未資格。旧SOURCE不変、実H4科学call0。

## 2026-10-27：PR-aware DF C2-A・whole-P0構造と未資格上界の技術準備

[固定ZSXX graph・条件付き回数制限・人工/回帰110 PASS](../../docs/research/research_c_pr_aware_df_c2a_synthesis_readiness_20261027.md)／[実行不可の未承認草案](../../docs/research/research_c_pr_aware_df_c2a_contract_draft_20261027.json)。新`c_pr_aware_df_c2a.py`／passive-only `check_research_c_pr_aware_df_c2a.py`／同名management test。Basis閉包等の前提下で採用rewrite≤N・採用matrix入力≤6Nを構成。Uniform ZSXX/initial translation/reference enclosureは未資格、科学call上限null・executable=false。SOURCE93/input23と固定Qiskit14SOURCEをpassive確認、新実H4科学call0、旧証拠保全、公開照合後mandatory STOP。


## 2026-10-27：PR-aware DF C2・Reference Bridge統合準備

新`trottertracks.c_pr_aware_df_c2_bridge`は既存CAR/general-K/K0 eventと保存B1 referenceを接続し、条件付きcategorical・Bridge・synthesis・reference誤差のraw単位schemaを用意する。[技術報告・人工/回帰85 PASS](../../docs/research/research_c_pr_aware_df_stage_c2_bridge_synthesis_readiness_20261027.md)／[未承認47-call契約](../../docs/research/research_c_pr_aware_df_stage_c2_bridge_contract_draft_20261027.json)。旧library/科学SOURCE不変、新実H4科学call0、native/C1再解析なし。全law P0・reference enclosure・PCG64 IIDは未資格、保証shots/totalはnull、mandatory STOP。

## 2026-10-27：PR-aware DF C2・全law資格の技術調査／未承認C2-B準備

[NumPy/P0 SOURCE・資格の限界・人工99 PASS](../../docs/research/research_c_pr_aware_df_stage_c2_qualification_readiness_20261027.md)／[未承認35-call表解析契約](../../docs/research/research_c_pr_aware_df_stage_c2_contract_draft_20261027.json)。`c_pr_aware_df_stage_c2.py`／静的`c_pr_aware_df_stage_c2_audit.py`／`run_research_c_pr_aware_df_stage_c2.py`／同名management test。Exact finite-grid TVと条件付きraw-law伝播、scoped synthesis／support入力、null C1-previewを人工検証。現whole-P0合成・PCG64 IID／CAR-atom-reference bridge・費用supportは未資格。SOURCE76/input12 passive PASS、新規実分子科学call0、C1再解析0。保存H4・CDF8二arm・元順LD2・δ.1/s4/r16/K0/m0..12・own-oracle/状態準備除外を維持。別科学許可後のみC2-B実行、公開照合後mandatory STOP。


## 2026-10-27：PR-aware DF Stage C・保存証拠からshots／serial資源への技術準備

[2経路・資格不足・人工73 PASS・C1/C2設計](../../docs/research/research_c_pr_aware_df_stage_c_readiness_20261027.md)／[未承認C1・164-call案](../../docs/research/research_c_pr_aware_df_stage_c1_contract_draft_20261027.json)。新`c_pr_aware_df_stage_c.py`／専用runner／人工test。Evidence laneは全law合成・実装law不足でnull、2明示scenarioで既存accuracy/Hoeffding/serial-costへ接続する案。SOURCE60/saved408のpassive照合PASS、旧1361ファイル不変。実H4新shots/総費用/再解析・新科学call0。C2はC1後の選択を要する非実行草案。公開照合後mandatory STOP。

## 2026-10-27：PR-aware DF B2・64 native kernel校正完了

[全gate・費用6指標・不確かさ・技術再実行](../../docs/research/research_c_pr_aware_df_stage_b2_execution_20261027.md)／[修正契約](../../docs/research/research_c_pr_aware_df_stage_b2_compat_contract_20261027.json)。保存H4・1Å/STO-3G、CDF8 Baseline/Proposed、元順LD2、partial S2 δ.1/s4/h1/40/r16/K0、own-DF oracle/状態準備除外、共通precision/P0。Run01は429 unitsでQiskit Already borrowed技術STOP（数値FAIL0）を保全し、共有build/compile/adjoint mutex・元64seedのfresh run02が652/652 COMPLETE／FAIL0、累積1081、技術再起動1・run02 retry/resume0。64 full512/adjoint/X-Yと67 compileがPASS。RZ標本mean±SEは5198.3125±66.7146対4765.875±80.3845（今回約8.3188%低い）。母平均保証・support・CA/shots/総資源は未資格。SOURCE50/input22/旧628不変、公開照合後mandatory STOP。

修正entryは`run_research_c_pr_aware_df_stage_b2_compat.py`、moduleは`c_pr_aware_df_stage_b2_compat.py`、testは同名management test。旧B2 runner/sourceは不変。

## 2026-10-27：PR-aware DF B2・native費用校正の技術準備

[共通native・full512・人工72 PASS・費用資格と制限](../../docs/research/research_c_pr_aware_df_stage_b2_preparation_20261027.md)／[未承認652-call契約](../../docs/research/research_c_pr_aware_df_stage_b2_contract_draft_20261027.json)。新`c_pr_aware_df_stage_b2.py`／専用runner／人工test。32独立trajectory/arm、64 kernel＋3shell、67 P0 compile、全operator/adjoint/X-Y後にwhole-kernel metricsを記録する案。SOURCE47/input22のbyte/header preflight PASS、旧175ファイル不変。実H4 decode/sampling/build-compile/cost・新科学call0、CI/support/ideal費用/CA/総資源は未資格。公開照合後mandatory STOP。

## 2026-10-27：PR-aware DF Stage Bの接続・人工検証準備

[2 CDF arm／law／sector reference／共通native接続・人工56 PASS](../../docs/research/research_c_pr_aware_df_stage_b_readiness_20261027.md)／[未承認B1-only141-call案](../../docs/research/research_c_pr_aware_df_stage_b_contract_draft_20261027.json)。新`c_pr_aware_df_stage_b.py`／`stage_b_runner.py`／同名runner・management test。保存Stage A J/Uの新評価は0、hash/header/JSON preflightのみPASS、旧97ファイル不変。B1は2arm×13roundのPF/RTE/A/signalをsector36で検証し、build/compile/sampling/shots/costは0の案。B2は人工native部品まで、実行範囲は別固定。科学call0、公開照合後mandatory STOP。

## 2026-10-27：PR-aware DFのx0接続を技術修正

[単一qualified x0引渡し・人工61 tests＋29 subtests](../../docs/research/research_c_pr_aware_df_x0_repair_20261027.md)／[新V2・未承認30-call契約](../../docs/research/research_c_pr_aware_df_pilot_contract_x0_v2_unapproved_20261027.json)。既存corrected-codecへopt-in prepared初期値を接続し、fit内の再logを除去。旧default・目的／gradient／optimizer条件は維持。実分子decode・新科学call0、旧18-call STOPを保全、新2026-10-27/run01未作成。別明示許可までmandatory STOP。

## 研究C：PR-aware DFの初期設計・技術準備（依頼識別子20261026）

[設計・再利用・人工49 tests＋29 subtests](<../../docs/research/research_c_pr_aware_df_optimization_design_20261026.md>)／[未承認Stage A・30-call案](<../../docs/research/research_c_pr_aware_df_pilot_contract_draft_20261026.json>)。`c_pr_aware_df.py`／`c_pr_aware_df_pilot.py`、`run_research_c_pr_aware_df.py`、同名management tests。既存ffsim loss/gradient・修正codec・SectorCARへ固定tail mass代理目的を最小接続。実optimizerは人工mock nit0、両env byte preflight PASS。実分子decode・新科学call0、科学的有効性／新規性・Stage B/Cは未資格・未承認、mandatory STOP。

## C1：H4 XDF9費用screen・両sでstrict改善を除外（2026-10-26）

[全gate・metrics・shots・比較・制約](../../docs/research/research_c_c1_h4_xdf9_cost_screen_execution_20261026.md)／[raw48・QPY4・83-call journal・STOP・CSV](../../artifacts/research_c_c1_h4_xdf9_cost_screen/2026-10-26/)。固定SOURCE7328ecc9a…無変更、83/83 COMPLETE、correctness FAIL/inflight/retry/resume0。直鎖H4・1Å/STO-3G、Nα=Nβ2、元順XDF9、全frame S2 s1/s4、δ.1/m0..12、own-ground/preparation excluded、同precision/P0/shared/fenced serial。kernel RZ15720/各s。楽観的総RZ s1=144805110560/s4=530107748679はいずれも保存XDF8/CDF8より大きく、固定十分処方・実装・数値資格内でstrict改善を除外。full operator/actual合成誤差/CA/保証付きshots/最終費用は未資格・未評価。旧結果保全、公開照合後mandatory STOP。

## C1：H4 XDF9 native／楽観的shots screenの技術準備（2026-10-26）

[最小接続・単調性／ceil・限界・full512後続案](../../docs/research/research_c_c1_h4_xdf9_cost_screen_preparation_20261026.md)／[未承認83-call契約](../../docs/research/research_c_c1_h4_xdf9_cost_screen_contract_unapproved_20261026.json)／[SOURCE・人工／回帰97 PASS・passive証拠](../../artifacts/research_c_c1_h4_xdf9_cost_screen/2026-10-26/technical/)。新entry `run_research_c_c1_h4_xdf9_cost_screen.py`／module `c1_h4_xdf9_cost_screen.py`／同名management test。既存PrecisionBuilder/P0/solver/serial/rational楽観集計を再利用、XDF9 S2 s1/s4の2 kernelとPF26から固定処方の楽観的RZ下限を接続。実H4のdecode/build-compile/shots/cost・科学call0。下限は固定処方／serial実装内の数値screen、full operator／strict enclosure／actual CA・shots・資源優位ではない。SOURCE56/input24 byte preflight PASS、旧証拠不変、別明示許可までmandatory STOP。

## C1：H4 XDF9・表現／ground-alias／PF26がPASS（2026-10-26）

[全gate・全26 PF値・科学call・資格の限界](../../docs/research/research_c_c1_h4_xdf9_pf_execution_20261026.md)／[raw45・SOURCE・journal・STOP・CSV](../../artifacts/research_c_c1_h4_xdf9_pf/2026-10-26/)。固定SOURCE9a4cda44a…無変更、95/95 COMPLETE、correctness FAIL/inflight/retry/resume0。直鎖H4・1Å・STO-3G、Nα=Nβ2/sector36、保存OpenFermion元順prefix9、元g補正κ・核scalar・frame順、全frame S2 s1/s4、δ.1・m0..12、PRなし、own-XDF9 oracle/状態準備除外。保存DF予算PASSを再利用、表現10因子・sum・own-ground/alias PASS、両sとも13/13 PASSのPOTENTIALLY_FEASIBLE。最大PF誤差s1 .06843041535127227/s4 .004277093283650178。新fit/DF norm/native/合成/CA/shots/総費用0、27 protected filesと原packet不変。旧約27%／約12%比較を保全し、公開照合後mandatory STOP。以下は過去の準備・結果の履歴。

## C1：H2／旧H4／新H4／H6統合と次期XDF9準備（2026-10-26）

[保存16候補・baseline差分・SOURCE来歴](../../docs/research/research_c_c1_h2_h4_h6_integration_update_20261026.md)／[A:H4 XDF9・B:H6 CDF13の比較と未採択推奨](../../docs/research/research_c_c1_next_minimal_experiment_selection_20261026.md)／[人工78 tests＋29 subtests・技術準備](../../docs/research/research_c_c1_h4_xdf9_preparation_20261026.md)。H4約27%→約12%をrank単独の因果効果とせず、3系をサイズscaling実験と扱わない。新entry `run_research_c_c1_h4_xdf9_pf.py`／module `c1_h4_xdf9_pf.py`／同名management testは、既存CAR／一般K／PFを再利用する[未承認95-call契約](../../docs/research/research_c_c1_h4_xdf9_pf_contract_unapproved_20261026.json)の準備。original input／SOURCE24／保存JSON3のbyte preflight PASS、実分子decode／新科学call0。旧証拠保全、推奨は未採択、公開照合後mandatory STOP。後続native／CA／shots／費用は別範囲。 [CSV・来歴](../../artifacts/research_c_c1_integration_update/2026-10-26/)／[SOURCE・人工log・passive証拠](../../artifacts/research_c_c1_h4_xdf9_pf/2026-10-26/technical/)。

## C1：H4保存証拠から既存RPE/総費用への数値pilot完了

[結果・全gate・shots/energy/総費用](../../docs/research/research_c_c1_h4_ca_resources_execution_20261026.md)／[raw19・52round/104軸CSV・SOURCE/calls/STOP](../../artifacts/research_c_c1_h4_ca_resources/2026-10-26/)。固定SOURCE18045333…無変更、119/119 COMPLETE、FAIL/inflight/retry/resume0。直鎖H4・1Å・STO-3G、CDF8/保存元順XDF8、Nα=Nβ2/sector36、全frame S2 s1/s4、δ.1・m0..12、PRなし、own-DF oracle/状態準備除外。4候補の共通CA=.00159 Ha十分条件が数値PASS。CDFの条件付き総RZ削減は同sでs1 11.853846%／s4 11.990819%。保存level B・ideal IID Born・固定serial native modelのpilotで、厳密CA・無条件優位ではない。新fit/DF-PF-合成誤差再計算/build-compile/量子shots抽選0、旧証拠保全・mandatory STOP。以下は実行前準備などの履歴。

## C1：H4保存full512から既存RPE/費用への接続

[既存再利用・人工/回帰88 PASS・資格の限界](../../docs/research/research_c_c1_h4_ca_resources_preparation_20261026.md)／[未承認119-call契約](../../docs/research/research_c_c1_h4_ca_resources_contract_unapproved_20261026.json)。新`c1_h4_ca_resources.py`／同名runner・testが、保存52 full512 profile・各DF norm＋guard・共通precision/P0 metricsから既存RPE/Hoeffding/energy/serial費用を接続。直鎖H4・1Å・STO-3G、CDF8/保存順XDF8、Nα=Nβ2/sector36、全frame S2 s1/s4、δ.1・m0..12、PRなし、own-DF oracle/状態準備除外。科学call0、SOURCE24/input12のbyte preflight PASS。新shots/CA/総費用は未評価、厳密enclosure・無条件優位は未資格。旧証拠保全・別明示科学許可までmandatory STOP。

## C1：H4共通precision・full512の実行結果

[結果・全gate・7 metrics・52 profile](../../docs/research/research_c_c1_h4_native_fair_execution_20261026.md)／[SOURCE・QPY40・call journal・STOP](../../artifacts/research_c_c1_h4_native_fair/2026-10-26/run01/)。SOURCE862a97d…を無変更実行し305/305 COMPLETE、correctness FAIL/retry/resume0。直鎖H4・1Å・STO-3G、CDF8/保存元順XDF8、Nα=Nβ2/sector36、全frame S2 s1/s4、δ.1・m0..12、PRなし、own-DF oracle/状態準備除外。同H6 Gaussian/P0・既存shared routeで局所18/kernel4/shell3/adjoint8/X-YがPASS、full512数値Frobeniusによる52/52位相条件内。単一h-step RZはCDF12,272対XDF13,944（約11.99%減）。丸めenclosure・共通CA・shots・総資源は未資格/未評価。旧SOURCE/raw/STOPを保全しmandatory STOP。以下の技術準備欄は実行前時点の記録。

## C1：H4共通precision・全Fock Frobeniusの最小接続

`../trottertracks/c1_h4_native_fair.py`は既存NativeBuilder/prepare_cdf/prepare_xdf、H6 Gaussian・MPFR P0、full CAR/Operator/QPY/metricsを無変更再利用。`c1_h4_full_synthesis.py`は既存Frobenius/reference_info/repeat_boundへ両行列・shell norm growthを接続。[技術報告・156 unique人工/回帰PASS](../../docs/research/research_c_c1_h4_native_fair_preparation_20261026.md)。未承認305-call scope、実H4新科学0、厳密enclosure/CA/総資源は未資格・mandatory STOP。

## C1：H4保存XDF8・sector表現/ground/alias/S2 PF実行完了

SOURCE328f31007…のadapterから既存matvec／CAR／一般K／H4 eigensolver／PFを無変更再利用。[結果・93 COMPLETE・全26 PF PASS](../../docs/research/research_c_c1_h4_xdf8_pf_execution_20261026.md)。元順prefix8のs1/s4とも潜在適格。新DF分解／fit／DF norm／回路／CA／shots／総費用なし。旧SOURCE/raw/STOP不変、mandatory STOP。

## C1：H4保存順XDF8の表現・ground/alias・PF adapter

`../trottertracks/c1_h4_xdf8_pf.py`は既存prefix matvec／SectorCAR／一般K／H4 eigensolver・alias／PF_rounds／call journalを再利用し、元順8frameのsector36評価へ接続する。[技術報告・新22＋回帰92 PASS](../../docs/research/research_c_c1_h4_xdf8_requalification_preparation_20261026.md)。新CAR／PF法／fitterなし、旧科学SOURCE不変。未承認93-call scope、実H4数値評価0・mandatory STOP。

## C1：保存H4 XDF prefix・限定DF予算評価完了

既存DF／sector／SVD／allowanceをSOURCE f7bc730d1…のadapterから無変更で再利用し、356 callsで全9prefixを評価。[結果・元順rank8が最初のDF予算PASS](../../docs/research/research_c_c1_h4_xdf_prefix_budget_execution_20261026.md)。新physical／fit／PF／回路／shots／費用なし。旧SOURCE・結果を保持、mandatory STOP。

## C1：保存H4 XDF prefix評価adapter

`../trottertracks/c1_h4_xdf_prefix_budget.py`は既存DFHamiltonian／PhysicalSector／df_linear_operator／dense化／bit permutation／SciPy SVD／H4 allowanceを再利用。薄いcounted matvecで全9prefix×36基底を計上する。[技術報告・人工29＋旧関連28 PASS・制約](../../docs/research/research_c_c1_h4_xdf_prefix_budget_preparation_20261026.md)。新sector simulator／fitterなし、旧SOURCEは不変。科学実行は別許可後だけ・mandatory STOP。

## C1：H6の共通CA・RPE shots・serial資源pilotを実行

既存accuracy/Hoeffding/fixed-N/energy/serial-cost関数をSOURCE55b3d868…のadapterから無変更再利用。119 calls・4候補数値CA PASS・同sの総RZはXDFが小さい。[結果・全指標](../../docs/research/research_c_c1_h6_ca_resources_execution_20261026.md)。資格B・理想測定/oracle/状態準備除外/serialモデルの条件付き結果であり厳密Cや無条件資源優位ではない。mandatory STOP。

## C1：H6共通CA・RPE shots・serial資源の最小adapter

`../trottertracks/c1_h6_ca_resources.py`は既存AccuracyTarget/assess_round/assess_energy、Hoeffding、固定N検証、compose_serial_cost/aggregate_totals/compare_totalsを変更せず再利用。[報告](../../docs/research/research_c_c1_h6_ca_resources_preparation_20261026.md)。保存combinedとの同一性と資格B、candidate別failure0.01、予定測定/serial費用を接続。人工/回帰102 PASS、実H6評価0・mandatory STOP。

## C1：H6部分空間の数値合成上界を取得

`c1_subspace_synthesis_bound.py`／`c1_h6_synthesis_bound.py`をSOURCE e36e056…無変更で実行し、全800列・full8192出力の資格Bと104roundの位相条件PASSを取得。[報告](../../docs/research/research_c_c1_h6_synthesis_bound_execution_20261026.md)。新fit/build/compile/DF-PF再計算0、厳密C/CA/shots/総費用は未資格、mandatory STOP。

## C1：H6合成誤差の全sector基底上界

`../trottertracks/c1_subspace_synthesis_bound.py` は全入力列のFrobenius上界と不変部分空間の反復伝播を実装。`c1_h6_synthesis_bound.py` は既存Aer/QPY/CAR列順・保存PF/shellへ接続する。[報告](../../docs/research/research_c_c1_h6_synthesis_bound_preparation_20261025.md)。資格Bを準備しCへ昇格しない。人工/回帰134 PASS、実H6は未実行・mandatory STOP。

## C1：H6 native修正版・実分子の限定検証完了

H6専用Gaussian精度／declared compiler phase修正をSOURCE3d7b56b448…で実行し、298 calls・局所30/kernel8/shell3/adjoint8/X-Y8全PASS。[結果](../../docs/research/research_c_c1_h6_native_stable_phase_execution_20261025.md)。全operatorの厳密enclosureは得ていない。CDF14のkernel RZはXDF14より約2%多く、literal/shared差0。global合成誤差/CA/総費用は未資格・mandatory STOP。

## C1：H6 declared compiler phase集計

`../trottertracks/c1_h6_compiler_precision.py`は既存P0 pass・gate本体・角度を保ち、global_phaseの加算だけ既存SymEngine MPFR160で安定化する。出力state/operatorのphase fitは行わない。[97人工PASS・修正報告](../../docs/research/research_c_c1_h6_native_stable_phase_repair_20261025.md)。

## C1：H6限定Gaussian数値精度修正

`../trottertracks/c1_h6_native_precision.py`は既存Nature JW消去順序・RZ/XXPlusYYを維持し、小回転をexact-zero/LAPACK/atan2で保持する。H6のbuilderだけ差替え、旧H2/H4 builderは変更しない。`c1_h6_native.py`はP0/inverse生成の共通mutexを使用。[報告・185人工PASS](../../docs/research/research_c_c1_h6_native_precision_repair_20261025.md)。

## C1：H6 native実行・部分証拠

`c1_h6_native.py`／`c1_h6_state_action.py`をSOURCE bf431765…から無変更実行。局所full8192有限probe30／ancilla shell3 PASS、kernel raw/referenceの最初の差1.0400731857e-10>固定1e-10で停止。162/298 units、kernel bundle8 FAIL、kernel費用・adjoint/X-Y未到達。[結果](../../docs/research/research_c_c1_h6_native_execution_20261025.md)。全operator/合成誤差上界/CA未資格、retry0、mandatory STOP。以下は準備履歴。

## C1：H6 native state-action接続

`../trottertracks/c1_h6_native.py`は既存NativeBuilder/一般K/CAR/P0/metricsを再利用。`c1_h6_state_action.py`は最大3-qubit matrixと既存Aer CPUでfull8192状態を作用させる。有限probeをoperator上界へ昇格させない。[技術報告](../../docs/research/research_c_c1_h6_native_preparation_20261025.md)。117 tests＋29 subtests PASS、実H6新科学は未承認、mandatory STOP。

## C1：保存H6 rank14の一般K・S2 PFを限定検証

`c1_h6_saved_pf.py`と既存SectorCAR/一般K/PF_roundsをSOURCE87ea924ae…で無変更実行。72 calls、一般K・ground/alias・全52PF条件PASS。4候補は潜在適格でCAや総費用は未資格。[結果](../../docs/research/research_c_c1_h6_saved_pf_execution_20261025.md)。mandatory STOP。

## C1：保存H6 rank14の一般K・S2 PF接続

`../trottertracks/c1_h6_saved_pf.py` は既存SectorCAR／一般K／PF_roundsを再利用し、保存CDF14/XDF14の4候補・52セルへ接続する。保存DF誤差を再計算せず、新fitやnativeを呼ばない。[技術報告](../../docs/research/research_c_c1_h6_saved_pf_preparation_20261025.md)。人工85 tests＋29 subtests PASS、科学実行は別途許可後のみ。

## C1：H6修正codecの実ffsim接続をrank14で限定検証

SOURCE33432e786…の`c1_ffsim_corrected_codec.py`／`c1_h6_corrected_fit.py`を無変更で実行。1 fit、20 calls、optimizer成功・DF予算PASS。初期tensor改善とsector悪化を区別する。[実行報告](../../docs/research/research_c_c1_h6_corrected_fit_execution_20261025.md)。後続科学は未実施、mandatory STOP。

## C1：H6修正codecのffsim最適化接続

`../trottertracks/c1_ffsim_corrected_codec.py` は専用process内で保存XDF初期値と修正codecを既存ffsim fitterへ接続し、loss/gradient/minimizeを維持する。`c1_h6_corrected_fit.py` は開始sector gateと終了optimizer／予算を分離する。新規fitは未承認。[技術報告](../../docs/research/research_c_c1_h6_corrected_fit_preparation_20261025.md)。

## C1：実直交DF codecとH6限定接続

[`c1_real_orthogonal_codec.py`](../trottertracks/c1_real_orthogonal_codec.py)はSciPy real Schur/expmとffsim parameter utilのtask-local接続。[`c1_h6_codec_roundtrip.py`](../trottertracks/c1_h6_codec_roundtrip.py)は既存SectorCAR/rank_assessmentを再利用する評価専用経路。[技術報告](../../docs/research/research_c_c1_h6_codec_roundtrip_preparation_20261025.md)。既存モジュール無変更、人工134 tests＋29subtests PASS、実H6新評価未実行・mandatory STOP。

## C1：保存H6初期値変換の限定診断

[`../trottertracks/c1_h6_initialization.py`](../trottertracks/c1_h6_initialization.py)は既存ffsim pack/unpack・JAX value/gradientとSectorCAR／rank_assessmentを接続する。新fitter・Gaussian合成器なし、列符号案は人工検証のみ。[技術・科学結果](../../docs/research/research_c_c1_h6_initialization_execution_20261025.md)、人工新12＋回帰28 PASS、35-call限定再解析完了・mandatory STOP。

関連trackの `../trottertracks/c1_molecule_extension.py` は[H2／H6への最小接続](../../docs/research/research_c_c1_h2_h6_preparation_20261025.md)。既存PySCF/ao2mo、ffsim bridge、PhysicalSector/excitation table、疎CAR、一般K、native/P0、PF/RPE/費用を再利用。H2はfull16／controlled32、H6はsector400だけでDF/PFを評価し、full4096／controlled8192のdense評価を行わない。SOURCE・人工結果 `artifacts/research_c_c1_h2_h6_technical/2026-10-25/`、CLI `scripts/run_research_c_c1_h2_h6.py`、test `tests/management/test_research_c_c1_h2_h6.py`。rank/guard policy・196/129-call案は未承認。実科学0・mandatory STOP。

関連trackの `../trottertracks/c1_stage3_deterministic_accuracy.py` は[決定論S2の精度資格付け準備](../../docs/research/research_c_c1_stage3_deterministic_accuracy_preparation_20261025.md)。既存CAR／local reference／RPE horizon・energy・費用集計を再利用し、保存4候補の固定Nを逆Hoeffdingで検査する最小拡張。s1保存Stage2 Frobenius、s4保存compiler差を再利用。entry `scripts/run_research_c_c1_stage3_deterministic_accuracy.py`、人工test `tests/management/test_research_c_c1_stage3_deterministic_accuracy.py`。SOURCE・人工結果 `artifacts/research_c_c1_stage3_deterministic_accuracy_technical/2026-10-25/`、未承認110-call契約。実H4科学call0、厳密reference enclosureは未資格・mandatory STOP。

`../trottertracks/c1_stage3_cost_compat.py`は旧cost-only backendを継承し、P0 compile callだけprocess内mutexへ接続する。旧数値／回路SOURCE・compiler profile・936 call・8候補を維持する。人工100 PASS、同時compile例外を人工再現し修正を確認。[修正報告](../../docs/research/research_c_c1_stage3_cost_compat_repair_20261025.md)、entry `scripts/run_research_c_c1_stage3_cost_compat.py`。

C1 Stage3の8候補費用経路は`../trottertracks/c1_stage3_cost_only.py`。旧native calibrator・P0・metric・serial期待費用・全support判定を再利用し、旧CAR/PF/RTE準備とshots再算定を避ける。entry `scripts/run_research_c_c1_stage3_cost_only.py`、test `tests/management/test_research_c_c1_stage3_cost_only.py`、[報告・未承認936-call契約](../../docs/research/research_c_c1_stage3_cost_preparation_20261025.md)。保存楽観Nでの費用は同CA保証ではない。人工94 PASS、実H4回路／総費用0・mandatory STOP。

C1 Stage3のshotsスクリーニングは`../trottertracks/c1_stage3_shots_screen.py`。旧精度geometry／Hoeffding／feasibility分類／call・partial証拠を再利用し、極小Aの任意精度countだけ最小拡張。全20候補を保持し、実装誤差が仮になければ式が要求する数を算定する。普遍的下界・CA保証ではない。entry `scripts/run_research_c_c1_stage3_shots_screen.py`、test `tests/management/test_research_c_c1_stage3_shots_screen.py`、[報告・未承認契約](../../docs/research/research_c_c1_stage3_shots_preparation_20261025.md)。人工81 PASS、保存値再解析0・STOP。

C1 Stage3の限定数値modeは`../trottertracks/c1_stage3_feasibility.py`。旧CAR／保存loader／ground・alias helper／PF・RTE reference／候補順序／call・資源計画を再利用し、QPY／回路／sampling／shot・費用経路へ接続しない。entry `scripts/run_research_c_c1_stage3_feasibility.py`、test `tests/management/test_research_c_c1_stage3_feasibility.py`、[技術報告](../../docs/research/research_c_c1_stage3_feasibility_preparation_20261025.md)、証拠 `artifacts/research_c_c1_stage3_feasibility_technical/2026-10-25/`。科学上限639、実H4未実行・STOP。


C1 Stage3の統合実行は`../trottertracks/c1_stage3_integrated.py`と`c1_stage3_molecular.py`、全law資格は`c1_stage3_qualification.py`、負係数RTE検査の新versionは`c1_stage3_native_bridge.py`。既存一般K／CAR／native／PF・RTE／RPE／compiler／費用評価を再利用し、旧SOURCEは保全する。entry `scripts/run_research_c_c1_stage3_integrated.py`、test `tests/management/test_research_c_c1_stage3_integrated.py`、[統合技術報告](../../docs/research/research_c_c1_stage3_integrated_preparation_20261025.md)、証拠 `artifacts/research_c_c1_stage3_integrated_technical/2026-10-25/`。実H4科学計算0、全lawの有用な合成誤差上界未資格、STOP。


C1 Stage3の新しい最小接続は`../trottertracks/c1_stage3_connection.py`（一般K tail／RTE／native直列接続）、`c1_stage3_reference.py`（独立generator／PF／RTE／virtual RPE）、`c1_stage3_accuracy.py`（CA／signal／shots／費用）にある。既存RTE・horizon・Hoeffding・phase復元・compilerを再利用。旧square-specific cost providerと短q proxyはCDFへ直接転用しない。[技術報告](../../docs/research/research_c_c1_stage3_preparation_20261025.md)、entry `scripts/run_research_c_c1_stage3.py`、人工test `tests/management/test_research_c_c1_stage3.py`。実H4科学計算なし、条件未採択・STOP。


関連trackの[C1 Stage2限定科学結果](../../docs/research/research_c_c1_stage2_native_execution_20261025.md)は既存Givens／diagonal emitter／CAR／controlled／X/Y／compilerを無変更で再利用し、全60主検査とunitarity52件PASS。CDF保存rank8／XDF rank10、call243/243、fit／retry／resume0。証拠 `artifacts/research_c_c1_stage2_science/2026-10-25/`。物理DF精度差があり同精度総資源優位ではない。Stage3未承認・mandatory STOP。

関連trackの[C1 Stage2 native接続](../../docs/research/research_c_c1_stage2_native_preparation_20261025.md)は `trottertracks.c1_stage2_native` から既存 `df_trotter.ops/decompose` と独立CAR、controlled/X/Y/compiler/metric資産を再利用する。旧library変更なし。entry `scripts/run_research_c_c1_stage2_native.py`、test `tests/management/test_research_c_c1_stage2_native.py`、SOURCE／証拠 `artifacts/research_c_c1_stage2_technical/2026-10-25/`。人工80件＋29 subtests PASS、科学条件は[未承認案](../../docs/research/research_c_c1_stage2_native_contract_unapproved_20261025.json)、実回路・科学call0、mandatory STOP。

関連trackの[C1保存fit評価専用科学実行](../../docs/research/research_c_c1_stage1_eval_only_execution_20261025.md)は `trottertracks.c1_stage1_eval_only` と既存CAR／一般K／XDFを無変更で使用した。full256 base・全8 frame／sector36を含む37 gate PASS、CDF/XDF物理誤差は別測定。entry `scripts/run_research_c_c1_stage1_eval_only.py`、人工19件PASS、科学call141/141、新fit／optimizer／retry／resume0。証拠 `artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/`。Stage2は別判断・mandatory STOP。以下の未承認・未再検証表示は当時の履歴。

関連trackの [C1保存fit評価専用準備](../../docs/research/research_c_c1_stage1_eval_only_preparation_20261025.md)は `trottertracks.c1_stage1_eval_only` から既存修正済みCAR／一般K／保存XDF評価を再利用し、数値libraryは無変更。entry `scripts/run_research_c_c1_stage1_eval_only.py`、新test `tests/management/test_research_c_c1_stage1_eval_only.py`、SOURCE／証拠 `artifacts/research_c_c1_stage1_eval_only_technical/2026-10-25/`。旧fit／新評価SOURCEを分離、人工82件＋29 subtests PASS、[未承認契約](../../docs/research/research_c_c1_stage1_eval_only_contract_unapproved_20261025.json)。実H4評価0、mandatory STOP。

関連trackの [C1 raw CAR frame修正](../../docs/research/research_c_c1_frame_repair_20261025.md)は `trottertracks.c1_stage1.raw_frame` の縮約だけをexplicit batched matmulへ接続。一般K／独立CAR代数／保存XDF／allowanceを保持し、新test `tests/management/test_research_c_c1_frame_repair.py` と既存回帰63件＋29 subtests、別プロセス64人工frame比較PASS。SOURCE／証拠 `artifacts/research_c_c1_frame_repair/2026-10-25/`、分子再解析・新fit0、H4の旧FAILは未再検証のままmandatory STOP。

関連trackの [C1 Stage 1互換性修正版実行](../../docs/research/research_c_c1_stage1_compat_execution_20261025.md)：SOURCE無変更で通常C-DF fit1回が収束（R8）。full256 frame0のraw CAR／K同値性がFAILし科学的STOP、既存保存XDF sector経路は構築したが物理CDF/XDF normは未到達。科学call107/147、retry/resume0、原本 `artifacts/research_c_c1_stage1_compat_science/2026-10-25/`、追加科学実行なし。

関連trackの [C1 Stage 1互換性修正](../../docs/research/research_c_c1_stage1_compatibility_20261025.md)は `trottertracks.c1_ffsim_compat` のtask専用batch logm bridgeだけを追加し、既存一般K／独立CAR／保存XDFの数値経路を保持する。runner `scripts/run_research_c_c1_stage1.py`、人工tests `tests/management/test_research_c_c1_ffsim_compat.py` と既存回帰59件＋29 subtests PASS、SOURCE／lock／証拠 `artifacts/research_c_c1_stage1_compatibility/2026-10-25/`。[新未承認契約](../../docs/research/research_c_c1_stage1_compatibility_contract_unapproved_20261025.json)、分子科学call0、別承認待ち・mandatory STOP。

関連track `trottertracks.c1_stage1*` の[Stage 1限定科学実行](../../docs/research/research_c_c1_stage1_execution_20261025.md)はffsim/SciPy batch logm非互換でfit初期化中に停止。保存XDFの既存sector演算・CAR/K/物理normへは未到達。artifact `artifacts/research_c_c1_stage1_science/2026-10-25/`、科学call2・fit1 attempt、追加実行なし、mandatory STOP。

関連trackの `trottertracks.c1_stage1` / `c1_stage1_execution` は、既存 `df_hamiltonian` のsector演算を保存XDFに再利用する[C1 Stage 1技術準備](../../docs/research/research_c_c1_stage1_technical_preparation_20261025.md)。runner `scripts/run_research_c_c1_stage1.py`、人工tests `tests/management/test_research_c_c1_stage1*.py`、SOURCE/lock/証拠 `artifacts/research_c_c1_stage1_technical/2026-10-24/`。実分子科学実行・回路は未実行・未承認。

このディレクトリが実装本体である。`scripts/` は主にここにある関数を呼び出す実行入口、
`tests/` は回帰検査である。研究全体の入口は[`../../PROJECT_MAP.md`](../../PROJECT_MAP.md)を参照する。

## 現行研究の中心実装

### Hamiltonian・DF分割・基底状態

- `chemistry_hamiltonian.py`、`df_hamiltonian.py`：分子HamiltonianとDF表現
- `df_partial_randomized_pf.py`：DF分割、fragment変換、state-action PF係数計算の共通基盤
- `df_trotter/`：決定論DF回路とstate-actionの低レベル実装
- `df_gpu_statevector.py`：大きめの系を想定したstate-action支援
- `io_cache.py`：入力・計算結果の保存と読込み

### partial-$S_2$・finite RTE

- `df_partial_s2.py`：決定論half sweepとRTE中央部を持つpartial-$S_2$
- `df_partial_s2_repeated.py`：partial-$S_2$の反復
- `df_rte_tail.py`：ランダム側HamiltonianのI/Z/ZZ表現
- `df_rte_circuit.py`、`df_rte_qiskit.py`：RTEイベント回路構築
- `rte.py`：有限RTEの確率、打切り、誤差計算の基礎

### コンパイル後回路コスト

- `rte_compiled_cost.py`：イベント列の構築、コンパイル、metric集計
- `df_partial_s2_cost.py`、`df_partial_s2_repeated_cost.py`：partial-$S_2$のcost集計
- `rte_connected_cluster_cost_validation.py`：局所境界補正の較正・予測・独立検証
- `rte_boundary_cost_validation.py`、`rte_boundary_pair_validation.py`：境界補正の基礎検証
- `rte_order_stratified_cost_validation.py`：Taylor次数の条件付き抽出
- `rte_system_size_cost_validation.py`：系サイズ方向の検証
- `rte_cost_angle_invariance_validation.py`：回転角とmetric cacheの診断

### RPE

- `rpe_hadamard_interrogation.py`：X/Y Hadamard interrogation回路
- `df_rpe_resource.py`、`rpe_resource_accounting.py`：誤差・shot・1 shot costの資源集計
- `rpe_short_round_optimization.py`：短いRPE段の候補選択
- `df_rpe_hadamard_compiled_cost.py`：Hadamard回路のコンパイル後cost
- `rpe_hadamard_compiled_cost_proxy.py`：反復数方向のcost proxy
- `rpe_hadamard_validated_proxy_provider.py`：検証範囲を守ってproxyを供給

## 共有サーバー向け実行基盤

- `parallel_validation_executor.py`：決定論的task ID、資源guard、bounded subprocess、
  atomic checkpoint、resume、GPU job割当、決定論的集約

科学ロジックや科学的判定は含めない。運用方法と安全上の既定値は
[`../../docs/server_parallel_validation_execution.md`](../../docs/server_parallel_validation_execution.md)を参照する。

## 検証モジュール

次のモジュールは、研究用APIそのものではなく、条件固定、比較、判定、成果物生成を担当する。

- `finite_rte_signal_validation.py`
- `pf_delta_validation.py`
- `pf_c_system_size_validation.py`
- `pauli_partial_cgs_validation.py`
- `random_circuit_cost_validation.py`
- `hierarchical_cost_validation.py`
- `rpe_round_cost_connection_validation.py`
- `rpe_hadamard_failure_validation.py`
- `rpe_hadamard_proxy_resource_validation.py`
- `rpe_allocation_sensitivity_validation.py`
- `rpe_four_round_accounting_validation.py`
- `rpe_four_round_phase_validation.py`
- `rpe_target_round_horizon_validation.py`
- `rpe_delta_round_schedule_validation.py`
- `rpe_delta_compiled_cost_validation.py`
- `rpe_hadamard_compiled_cost_benchmark.py`

同名のrunner、test、文書、artifactと合わせて読む。モジュールが実装済みでも、対象範囲が
科学的に検証済みとは限らない。

## 解析モデルと旧経路が同居するモジュール

- `partial_randomized_pf.py`：誤差配分・解析cost modelの共通関数と旧Pauli screening経路
- `df_screening_cost.py`、`df_cost_plotting.py`：DF screening・可視化

これらは現行コードから共通関数を利用する場合がある一方、保存済みの旧screening結果は
失効している。モジュール全体を旧式とみなすのでも、既存出力を現行結果とみなすのでもなく、
利用する関数とartifactのstatusを個別に確認する。

## 旧高次PF・比較用実装

次はプロジェクトの出発点である高次積公式、UWC、旧screening経路を支える。現行の
DF部分ランダム化研究と混同しない。

- `optimal_trotter.py`、`product_formula.py`、`pf_decomposition.py`
- `Almost_optimal_grouping.py`
- `qiskit_time_evolution_grouping.py`、`qiskit_time_evolution_ungrouped.py`
- `qiskit_time_evolution_pyscf.py`、`qiskit_time_evolution_utils.py`
- `plots_timeevo_error.py`、`cost_extrapolation.py`、`rz_layers.py`
- `uwc.py`、`grouped_uwc_comparison.py`、`grouped_uwc_theta_sweep.py`

これらの結果を引用するときは、先に
[`../../VALIDATION_STATUS.md`](../../VALIDATION_STATUS.md)で失効・欠落状態を確認する。

## 共通支援

- `config.py`：共通設定
- `analysis_utils.py`、`plot_utils.py`：解析・可視化支援
- `matrix_multiply.py`、`matrix_pf_build.py`：行列ベースの参照計算
- `eig_error.py`、`qpe_beta.py`：固有値誤差・位相推定の旧来支援

新しい機能は、runnerへ直接大きな処理を書くのではなく、ここへテスト可能な関数として置く。


## PR混在辞書の保存入力adapter（F1-PRE）

既存 `rte.py`／`df_trotter/ops.py` を変更せず、`trottertracks/pr_dictionary_validation/f1_saved_execution.py` と `f1_saved_plan.py` から再利用する準備を追加した。[準備記録](../../docs/research/pr_algorithm_improvement_phase2d_f1_preparation_20261023/README.md)がrunner、管理test、planned artifactの入口。科学実行は未承認・未実行。


## PR混在辞書F1の部分検証（2026-10-24）

既存RTE／Gaussian helperを再利用したone-shotの[結果](../../docs/research/pr_algorithm_improvement_phase2d_f1_execution_20261024/README.md)。4arm H-ID等は部分的にPASSしたが、compile cache用metadata設定で停止。`trottertracks/pr_dictionary_validation` の固定SOURCEは保全し、retry／resumeなし。要約／hash台帳は `artifacts/pr_algorithm_improvement_phase2d_f1/2026-10-24/`。


## PR混在辞書F1：修正版の登録検証完了（2026-10-24）

[新結果とSTOP](../../docs/research/pr_algorithm_improvement_phase2d_f1_rerun_20261024/README.md)。既存saved runner／adapter／reference／pricingを再利用し、SOURCE `8192530` と保存B入力で16familyを全件完了。全4armの登録正しさ検査とcatalog価格-DP照合はPASS、未資格intervalはfallbackのみ。認証score／shot／資源優位性は未評価。artifactは `artifacts/pr_algorithm_improvement_phase2d_f1/2026-10-24-r1/`。旧停止記録は保全し、F2へ進まない。


## PR混在辞書 F2実行前準備（2026-10-24）

[準備・SOURCE再利用・case・統計資格・契約・STOP](../../docs/research/pr_algorithm_improvement_phase2d_f2_preparation_20261024/README.md)。入口 `scripts/run_pr_dictionary_f2_saved.py`、SOURCE `trottertracks/pr_dictionary_validation/f2_*.py`、管理test `tests/management/test_pr_dictionary_f2_preparation.py`。planned outputは `artifacts/pr_algorithm_improvement_phase2d_f2/<execution-date>/<fresh-run-id>/`。finite-target主比較を採択し、mock/管理確認のみ完了。統計条件とinterval資格は未解決で、科学実行・新grantは0件。


## PR混在辞書 F2統計資格確認（2026-10-24）

[資格確認と実行前STOP](../../docs/research/pr_algorithm_improvement_phase2d_f2_qualification_20261024/README.md)。既存interval／log／ceilの技術test `tests/management/test_pr_dictionary_f2_scalar_intervals.py` は5件PASS。科学SOURCEはF2-PREのまま、共通finite targetに対する有限ビット誤差予算γの保証と統計条件の採択が未解決で、F2科学実行・新grantは0件。機械可読記録は同資料の `qualification.json`。入口は既存 `scripts/run_pr_dictionary_f2_saved.py` を維持する。


## PR混在辞書 F2限定資源計測（2026-10-24）

[限定契約・再利用SOURCE・停止条件](../../docs/research/pr_algorithm_improvement_phase2d_f2_resources_20261024/README.md)。入口 `scripts/run_pr_dictionary_f2_resources_saved.py`、接続 `trottertracks/pr_dictionary_validation/f2_resources.py`、管理test `tests/management/test_pr_dictionary_f2_resources.py`。出力は `artifacts/pr_algorithm_improvement_phase2d_f2_resources/`。旧統計資格付き契約を維持し、認証Nを生成しない。

限定F2の[実行結果・部分資源・STOP](../../docs/research/pr_algorithm_improvement_phase2d_f2_resources_20261024/EXECUTION_REPORT.md)：raw全件PASS、compiled operator3件FAIL、mean／4arm期待資源は未完了。SOURCE `9b1e103` と一回の実行原本を保全し、retryなし。


## 限定F2・位相保持修正と新規実行（2026-10-24）

[修正・実行契約・停止条件](../../docs/research/pr_algorithm_improvement_phase2d_f2_resources_rerun_20261024/README.md)。既存runnerとf2_resources／f2_executionを再利用し、gates.compiled_wrapperの公式definition展開を修正。回帰testは `tests/management/test_pr_dictionary_compile_phase.py`、planned artifactは `artifacts/pr_algorithm_improvement_phase2d_f2_resources/2026-10-24-r1/`。

[修正版F2の全件結果・資源表・STOP](../../docs/research/pr_algorithm_improvement_phase2d_f2_resources_rerun_20261024/EXECUTION_REPORT.md)：SOURCE `9468c70`、raw／compiled各13,676・reference27,356・mean4arm・weighted集計を完了。1-shotの指標順はPP<SP<PS<SS、認証Nは対象外。旧失敗は保持し、追加実験へ進まない。

関連trackの `src/trottertracks/pr_dictionary_validation/f2_attribution.py` は[保存F2寄与分析](../../docs/research/pr_algorithm_improvement_phase2d_f2_attribution_20261024.md)。既存 `f2_execution.aggregate_metrics` を変更せず再利用し、分類だけを追加した。runnerは `scripts/run_pr_dictionary_f2_attribution_saved.py`、testは `tests/management/test_pr_dictionary_f2_attribution.py`、artifactは `artifacts/pr_algorithm_improvement_phase2d_f2_attribution/2026-10-24/`。回路・reference処理は呼ばない。

## 関連track：Rectangular GivensのPS48準備（2026-10-24）

`src/trottertracks/pr_dictionary_validation/rectangular.py` は既存Nature helperのGaussian suffixをsingletonにだけ接続する。[条件・SOURCE・未検証事項](../../docs/research/pr_rectangular_givens_preparation_20261024.md)。`rectangular_preflight.py`／`rectangular_execution.py`、runner `scripts/run_pr_rectangular_givens_saved.py`、mock test `tests/management/test_pr_rectangular_givens_preparation.py`、artifact `artifacts/pr_rectangular_givens_preparation/2026-10-24/`。共有runの既存合成を維持し、実回路・数値同値性・削減量は次回承認後の検証対象。

固定SOURCE `770e53a` の関連track `trottertracks.pr_dictionary_validation.rectangular*` による[PS48科学結果](../../docs/research/pr_rectangular_givens_ps48_execution_20261024.md)を取得した。6基底と全48 wrapperの同値性gateはPASS、資源差はcase依存。科学SOURCEの変更なし。結果は `artifacts/pr_rectangular_givens/2026-10-24/`、保存集計だけのutilityも同所。全F2や別候補へ進まずmandatory STOP。


## 研究C・C4：Q13限定科学検証（2026-10-24）

[実行契約・履歴・結果](../../docs/research/research_c_rustiq_execution_20261024.md)。SOURCEは `src/trottertracks/pr_dictionary_validation/rustiq_adapter.py` と `rustiq_execution.py`、入口は `scripts/run_research_c_rustiq_limited.py`、管理testは `tests/management/test_research_c_rustiq_*.py`、artifactは `artifacts/research_c_rustiq_limited/2026-10-24/`。保存F1の4case・B0/B1/R1だけを対象とし、旧SOURCE・grant・STOPを保持する。


## 研究C・C4：B2角度結合baseline限定科学比較（2026-10-24）

[結果・固定SOURCE・契約・証拠](../../docs/research/research_c_angle_fusion_b2_execution_20261024.md)。既存 `rustiq_adapter.py`／`rustiq_execution.py` と `scripts/run_research_c_rustiq_limited.py --scope C4_B2_SAVED_COMPARISON_V1` を使用。testは `tests/management/test_research_c_rustiq_*.py`、artifactは `artifacts/research_c_rustiq_angle_fusion/2026-10-24/attempt01/`。全登録gateがPASSし、全8wrapper・全8compiled指標でB2=保存R1。単純融合がこのfixtureの改善を再現した。旧route再実行・新reference・Rustiq/R2・retryは0。独立研究を区切るかはユーザー／研究C／統括の次判断であり未採択、mandatory STOP。

## H4強baselineのtask-local adapter（2026-10-26）

既存DF primitive／precision Gaussian／P0を再利用する`src/trottertracks/c1_h4_strong_baseline.py`を追加した。[技術報告](../../docs/research/research_c_c1_h4_strong_baseline_preparation_20261026.md)。exact spin XDF basisとanalytic vacuum phase付きrelative boundaryをopt-inで接続し、既存library/defaultは無変更。人工／回帰101 PASS、実H4科学実行0。s1三route・最大227-callの未承認案で、B3・shots・総費用へ自動進行しない。

## 2026-10-26：H4強baseline s1科学実行完了

[227-call結果・full512検証・5比較](../../docs/research/research_c_c1_h4_strong_baseline_execution_20261026.md)。9 local／3 kernel／6 adjoint／9 complex X/Y gate PASS。B1 XDF RZ11821、B2 CDF11062/XDF11307、保存B0は再buildなし。単一kernelの数値pilotで、B3・s4・CA/shots/総費用は未評価。旧結果・SOURCE・STOPを保全し、公開照合後mandatory STOP。

## H4 B3のtask-local接続（2026-10-26）

`src/trottertracks/c1_h4_b3.py` は旧StrongBuilderのSPINとrelative_fusionを併用する単一route接続。[技術報告](../../docs/research/research_c_c1_h4_b3_preparation_20261026.md)。Gaussian／Givens／DF／compiler／referenceの新実装なし、旧library/defaultは無変更。人工／回帰122 unique PASS、実H4新計算0。未承認65-call案の別科学許可までmandatory STOP。

## 2026-10-26：H4 B3 s1・65-call限定実行完了

[全結果・full512 gate・6指標・3比較](../../docs/research/research_c_c1_h4_b3_execution_20261026.md)。固定SOURCE b40fce5c…無変更、65/65 COMPLETE、correctness FAIL/retry/resume0。H4直鎖1Å/STO-3G、元順XDF8/保存CDF8、Nα=Nβ2、全frame S2 s1/h.1、PRなし、own-ground/状態準備除外、共通precision/P0。新B3 spin＋relative一件のみbuild/compileし、raw/P0/actual adjoint/X-Yは全512入力・出力で数値PASS。RZ10885/CX6892/totaldepth11973は保存B2 CDF11062/7008/12127より小さく、他3指標も小さい。単一kernel比較で、同CA総順位・13round・shots・費用・厳密enclosureは未評価／未資格。旧raw・SOURCE・STOP保全、mandatory STOP。上の準備・未実行表記は履歴。

## 2026-10-26：H4 B2 CDF／B3 XDFのCA・RPE・総資源の技術準備

[再利用・接続・人工76 PASS・資格の限界](../../docs/research/research_c_c1_h4_strong_baseline_ca_resources_preparation_20261026.md)／[未承認86-call契約](../../docs/research/research_c_c1_h4_strong_baseline_ca_resources_contract_draft_20261026.json)。新module `c1_h4_strong_ca_resources.py`／同名runner・testsが保存full512 segment・norm guard・shell・元PF13を既存propagate/RPE/energy/serial集計へ接続する。H4直鎖1Å/STO-3G、CDF8/XDF8、Nα=Nβ2、全frame S2 s1のみ・δ.1/m0..12、PRなし、own-ground/状態準備除外、共通precision/P0/CA.00159。回路／Operator再生成不要、実科学call0。旧SOURCE/raw/STOP・旧約12%総資源とB3単一kernel結果を保全し、新総順位・厳密enclosureは未評価／未資格。公開照合後mandatory STOP。

## 2026-10-26：H4強baseline CA・実行前binding修正

[原因・人工78 PASS・修正契約v2・実科学call0](../../docs/research/research_c_c1_h4_strong_ca_binding_repair_20261026.md)。旧86-call契約の保存contract参照が、同一bytesのdocs pathとartifact pathの文字列差でSTOPするため、canonical参照だけを修正した。科学SOURCE31件・数値条件・call上限86・旧契約/raw/STOP不変。run01未作成・未消費、科学実行は未開始。修正契約での別明示許可までmandatory STOP。

## 2026-10-26：H4強baselineの共通CA・serial総資源比較完了

[全結果・26round・shots・総費用・資格の限界](../../docs/research/research_c_c1_h4_strong_baseline_ca_resources_execution_20261026.md)。固定SOURCE1aeedd961…／binding-v2無変更、86/86 COMPLETE、correctness FAIL/retry/resume0。H4直鎖1Å/STO-3G、B2 CDF8/B3 XDF8、Nα=Nβ2、全frame S2 s1のみ・δ.1/m0..12、PRなし、own-ground/ideal IID Born/状態準備除外、共通precision/P0/CA.00159。両候補のCA数値十分条件PASS、shots13606/13600、条件付き総RZ102056444569/100267421030でXDFが1.75297%少ない。level B数値pilotの限定比較で、厳密enclosure・無条件／一般の優位は未資格。旧結果保全、公開照合後mandatory STOP。上の準備・未実行表示は履歴。
