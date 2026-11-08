# 文書索引

[C8-1保存4回路の数値修正結果](research/track_d_c8_1_q64_20261108/NUMERICAL_REPAIR_RESULT.md)：Linear H4/1Å/STO-3G/Nα=Nβ=2・元H/number-CDF8/元順LD2/S2/s1/r32/K0/h=.1、q64/cut32、元H厳密ground仮定・準備費除外。同じsample00/4 P0に対し、SOURCE46ce57e1で旧FAILの値を保存し、SOURCEf1e89590のsimulation fusion無効化で4 postcheck全PASS。固定精度不変、旧原本46項目/STOP保持、新sampling/raw/compile0。最終独立paired/χ総合gate・残7・n8価格統計・長q/全RPEは未実施、公開照合後mandatory STOP。

[C8-1保存4wrapperの数値修正PLAN](research/track_d_c8_1_q64_20261108/NOFUSION_REPAIR_PLAN.md)。残7 pathは対象外。

[C8-1数値修正PLAN](research/track_d_c8_1_q64_20261108/NUMERICAL_REPAIR_PLAN.md)：保存回路だけの承認済み再検証。

[Track D C8-1最終結果](research/track_d_c8_1_q64_20261108/REPORT.md)：Linear H4/1Å/STO-3G/Nα=Nβ=2・元H/number-CDF8/元順LD2/S2/s1/r32/K0/h=.1、q64/cut32、元H厳密ground仮定・準備費除外。全8 event/512 occurrence/16384 eventsを固定し、sample00 raw1 PASS・旧P0 compile4完了をSOURCE8a6bbdacへ承認済み引継ぎ。両方式X/Y全4 postcheckがCAR-observable Frobenius2e−10でFAILしSTOP。残7/追加28 compile未開始、価格統計/モデル診断未評価。精度緩和/再実行/fit/N/全RPE更新0、旧SOURCE/STOP・非公開原本保持。

[C8-1高速化SOURCE引継ぎ](research/track_d_c8_1_q64_20261108/ACCELERATED_COMPLETION_PLAN.md)：compileの過大CPU予約を修正し、旧task完了待ち後の未着手処理を継続。19人工/回帰PASS、sampling追加0、old raw1/compile4再利用。固定q64/cut32・精度・モデル・全32 compile上限維持。最終C8-1結果公開照合後mandatory STOP。

[Track D B/C全13-round条件付き評価](research/track_d_bc_total_20261108/REPORT.md)：同Linear H4/1Å/STO-3G/Nα=Nβ=2・元H/number-CDF8/元順LD2/S2/s1/r32/K0/h=.1、元H厳密ground・準備費除外。保存十分N14738と固定A/F/Sで全13-round条件付き資源会計を一回実施。主AはReferenceでCX11.17%・total depth54.65%・RZ1.12%減、幅9→17。約99.5%が長q proxy、全体不確かさ/actual未資格。新sampling/compile/fit/N0、SOURCEb09b7e7a、公開照合後mandatory STOP。

[Track D B6b＋B/C統合準備](research/track_d_b6b_bc_integration_20261108/REPORT.md)：Referenceのideal coherenceを任意入力へ導出し、whole-V-ray条件とDF半path/最終真空条件を区別。SOURCE13114b2b、人工/回帰24件と人工P0 16 wrapper postcheck PASS。B4-2保存Nの条件付き共通利用・13-round価格coverage・薄い会計adapterを整理。actual N/長q期待価格は未資格、実H4の新計算/価格再解析/N算定/全RPE数値会計0。公開照合後mandatory STOP。

[Track D C7-3結果](research/track_d_c7_3_q32_20261107/RESULT.md)：同H4/1Å/STO-3G/Nα=Nβ=2・number-CDF8/元順LD2/S2/s1/r32/K0/h=.1、独立q32/cut16/8 path。sampling/raw SOURCE35bfd、承認済み高速化compile/post SOURCE94e94。旧raw8再利用・新raw/sampling0、全32 P0/post・pairedχ8 PASS。CX差は全8本で−6960、RZ/size/各depthも有限全8本で減、qubits9→17。一次12量は探索的幅内、別population-line6量幅外。統計/全law/q≥64/N/全RPE未資格、旧STOP/raw非公開を維持、公開照合後mandatory STOP。

[C7-3高速化・切替PLAN](research/track_d_c7_3_q32_20261107/ACCELERATED_COMPLETION_PLAN.md)：利用者承認のSOURCE変更。旧raw8 PASSと旧STOPを保全し、未着手compile/postへ切替。case/model/精度/P0は維持、追加sampling/raw再実行0、人工/回帰61 PASS。

[Track D C7-3 PLAN](research/track_d_c7_3_q32_20261107/PLAN.md)：独立q32/cut16/8 path、凍結予測と新domainを事前固定。raw gate後最大32 full P0、C7-2構造feature、探索的価格のみ。q≥64/新fit/N/全RPE対象外、公開後mandatory STOP。

[Track D C7-2](research/track_d_c7_2_cx_structure_20261107/REPORT.md)／[q32案](research/track_d_c7_2_cx_structure_20261107/C7_3_EXECUTION_DRAFT.md)：固定54人工P0と局所CX/phase/DAG検証、全54 postcheck・回帰12 PASS。非加法/単純lane式へ反例。実H4の新計算・再解析0、全path/長q未資格、mandatory STOP。

[Track D C7-1](research/track_d_c7_1_q16_20261107/RESULT.md)：同H4/1Å/STO-3G/Nα=Nβ=2・number-CDF8/元順LD2/S2/s1/r32/K0/h=.1、独立q16/cut8/16 path。SOURCE462f6bff、raw16・post64・pairedχ16 PASS。CX差は全観測pathで−3376、RZ/size/各depthも有限全16本で減、幅9→17。一次12量は探索的幅内、別population-line4量幅外。統計/全law/q≥32/N/全RPE未資格、raw非公開、mandatory STOP。

[Track D C7-1実行前PLAN](research/track_d_c7_1_q16_20261107/PLAN.md)：独立q16/cut8/16 path、凍結モデル・新seed、raw gateと最大64 P0 wrapperの探索的paired検証。旧SOURCE/STOP保全、原本ローカル、公開照合後mandatory STOP。

[Track D C7-0構造解析](research/track_d_c7_0_structural_preparation_20261107/REPORT.md)／[C7-1以降の条件案](research/track_d_c7_0_structural_preparation_20261107/C7_1_EXECUTION_DRAFT.md)：保存q2/4/8とSOURCEの静的調査。CXの条件付き構造、RZ/size、DAG/lane、estimand/校正/biasと最小q16案。新科学計算・再解析・実装・テストなし、全path/長q/全RPE未資格、公開照合後mandatory STOP。

[Track D C6b](research/track_d_c6b_q8_20261107/RESULT.md)：同H4/1Å/STO-3G/Nα=Nβ=2・number-CDF8/元順LD2/S2/s1/r32/K0/h=.1/q8/cut4、独立domainの32 path。元H厳密ground仮定・準備費除外。SOURCE44a58b9fへ承認済み限定引継ぎ、raw32・postcheck128・pairedχ32 PASS。一次frozen条件付き12量は探索的幅内、別population-line診断4量は幅外。統計未資格、全law/長q/N/全RPE未評価、raw非公開、mandatory STOP。

[C6b高速化による残task限定引継ぎ](research/track_d_c6b_q8_20261107/ACCELERATED_COMPLETION_PLAN.md)：旧科学条件・eventを保持し、完了済みraw/compile/postcheckを再実行しない。追加sampling0、全32本/旧新合算128 compile、公開後mandatory STOP。

[Track D C6b独立q8](research/track_d_c6b_q8_20261107/PLAN.md)：32 fresh whole-path、凍結C6aモデル、paired12量の探索的価格診断。確認的PASS・長q・全RPE費用は対象外、rawローカル保全、公開後mandatory STOP。

[Track D C6b-0](research/track_d_c6b0_preparation_20261107/REPORT.md)／[C6b実行前契約案](research/track_d_c6b0_preparation_20261107/C6B_EXECUTION_DRAFT.md)：C6a固定保存値とSOURCEの静的調査だけ。paired優先・追加校正要否・q8事前条件を整理、未採択・未実行、mandatory STOP。

[C6a結果](research/track_d_c6a_paired_price_20261106/RESULT.md)：fresh q2/q4各16、raw32/P0 postcheck128/pairedχ PASS。有限pilot統計・共有校正共分散・C6b提案、rawローカル保全、mandatory STOP。

[Track D C6a固定PLAN](research/track_d_c6a_paired_price_20261106/PLAN.md)：q2/q4各16 fresh paired価格pilot、最大128 P0、C6b未実施。

[C5結果](research/track_d_c5_q2_20261106/RESULT.md)：共通request入口・人工16件・固定q2 raw/4postcheck PASS。strong手動境界融合/位相集約を統一し、P0後7価格は旧C2と一致。有限fixture限定、C6未実施、mandatory STOP。

[Track D C5：共通whole-path入口とq2公平比較](research/track_d_c5_q2_20261106/PLAN.md)：保存sample00→01/cut1、C3と同じ強い境界融合・位相集約、raw PASS後だけ最大4 P0 wrapper。C6や新価格標本は未許可。

[Track D C4：期待価格・長qモデルの技術準備](research/track_d_c4_price_model_preparation_20261106/REPORT.md)：強いwhole-path比較の費用境界、保存価格の条件差、校正・独立holdout・不確かさ・B側Nとのinterface。新集計/fit/sampling/回路生成/compile/実装は行わない静的報告。

[C3 q4/q8補完結果](research/track_d_c3_q4_q8_20261106/COMPLETION_RESULT.md)：raw両q・全8 postcheck・最終paired/χ照合PASS。強いstep境界融合を維持して両fixtureでCX/depth減、qubits+8。中断後は保存回路の最後の検証のみ、新compile0。固定fixture限定・mandatory STOP。

[Track D C3 q4/q8固定PLAN](research/track_d_c3_q4_q8_20261106/PLAN.md)：whole-path correctnessとstrong境界融合を維持した最大8 wrapper P0価格検証。q1/q2再実行・全law/RPE費用は対象外。

[C2固定q2結果](research/track_d_c2_q2_compile_20261106/RESULT.md)：4 wrapper compile/postcheck PASS。ReferenceはCX4100→3860・depth6096→3014、RZ/qubits増。固定fixture限定、mandatory STOP。

[Track D C2固定q2 compile](research/track_d_c2_q2_compile_20261106/PLAN.md)：C1の保存rawを使う2方式×X/Yの4 wrapper限定。P0・cut・phase固定、公開後mandatory STOP。

[B1/C1結果](research/track_d_b1_c1_validation_20261106/RESULT.md)：元H向け13-round条件付きoperator十分条件成立、固定q2 raw correctness PASS。compile・新sampling・N評価0、B2/C2未実施、mandatory STOP。

[Track D B1/C1固定検証](research/track_d_b1_c1_validation_20261106/PLAN.md)：保存13-roundの元H向けoperator誤差接続と、sample00→01・q2のwhole-path raw correctnessだけ。B2/C2・compile・新samplingは対象外。

[Track D Stage B・Cの技術準備](research/track_d_bc_static_preparation_20261106/REPORT.md)：B（元H向け信号・統計）、C（whole-round構成・価格）、共通の最小検証順序とSTOP。新規科学計算・実装を伴わない静的調査。

[Stage A結果](research/track_d_h4_paired32_reference_20261106/RESULT.md)：保存32本すべてでCX/depth減、RZ増、qubits9→17。32 precompile／128 postcheck PASS、有限cohort限定、mandatory STOP。

[Track D Stage A：保存32 trajectoryのpaired Reference検証](research/track_d_h4_paired32_reference_20261106/PLAN.md)：同endpoint・固定cut、全precompile PASS後だけ2方式×X/Y×32の最大128 wrapper比較。元H厳密ground仮定・準備費除外、B/C段階は対象外。

[Track D実H4 Reference移送](research/track_d_h4_reference_validation_20261106/PLAN.md)／[結果](research/track_d_h4_reference_validation_20261106/RESULT.md)：保存word02・固定cut、strong symmetric対照、correctnessと6 full-wrapper compileの限定scope。

[Track D人工Reference検証](research/track_d_minimal_reference_validation_20261106/PLAN.md)：固定1 wordのcorrectness、3方式×X/Yの6 wrapper比較、SOURCE・STOP範囲。

研究別の成果・SOURCE・raw・検証状態への入口は[研究結果への統一索引](research/RESEARCH_RESULTS_INDEX.md)。
このcheckoutにない後続資料も、固定commitから参照できる。

このディレクトリには、研究方針の正本、実装規約、検証報告、発表資料の案内が共存する。
研究全体を初めて読む場合は、先に[`../PROJECT_MAP.md`](../PROJECT_MAP.md)と
[`research/研究概要・現状.md`](research/研究概要・現状.md)を読む。

## 研究方針と現在地

- [`research/研究概要・現状.md`](research/研究概要・現状.md)：最新の短い全体要約
- [`research/README.md`](research/README.md)：研究文書内の索引
- [`research/研究目的・研究課題.md`](research/研究目的・研究課題.md)：目的と研究課題
- [`research/研究方法・解析手順.md`](research/研究方法・解析手順.md)：採用する解析手順
- [`research/数値実験・評価計画.md`](research/数値実験・評価計画.md)：検証と評価の計画
- `research/研究ノート/`：時系列の判断記録。現在の仕様ではない

## 現在の主な検証文書

### PF係数

- [`pf_delta_validation.md`](pf_delta_validation.md)
- [`pf_c_system_size_validation.md`](pf_c_system_size_validation.md)

### finite RTE

- [`rte_conventions.md`](rte_conventions.md)
- [`rte_truncation_budget.md`](rte_truncation_budget.md)
- [`finite_rte_signal_validation.md`](finite_rte_signal_validation.md)
- [`df_rte_tail_extraction.md`](df_rte_tail_extraction.md)
- [`df_rte_event_circuit_api.md`](df_rte_event_circuit_api.md)

### コンパイル後回路コスト

- [`random_circuit_cost_validation.md`](random_circuit_cost_validation.md)
- [`rte_boundary_cost_validation.md`](rte_boundary_cost_validation.md)
- [`rte_boundary_pair_validation.md`](rte_boundary_pair_validation.md)
- [`hierarchical_cost_validation.md`](hierarchical_cost_validation.md)
- [`rte_connected_cluster_cost_validation.md`](rte_connected_cluster_cost_validation.md)
- [`rte_compiled_cost_validation_summary.md`](rte_compiled_cost_validation_summary.md)
- [`rte_compiled_event_cost.md`](rte_compiled_event_cost.md)
- [`df_partial_s2_compiled_cost.md`](df_partial_s2_compiled_cost.md)
- [`df_partial_s2_repeated_compiled_cost.md`](df_partial_s2_repeated_compiled_cost.md)

### RPEへの接続

- [`rpe_resource_accounting.md`](rpe_resource_accounting.md)
- [`rpe_hadamard_interrogation.md`](rpe_hadamard_interrogation.md)
- [`rpe_round_cost_connection_validation.md`](rpe_round_cost_connection_validation.md)
- [`rpe_hadamard_failure_validation.md`](rpe_hadamard_failure_validation.md)
- [`rpe_hadamard_proxy_resource_validation.md`](rpe_hadamard_proxy_resource_validation.md)
- [`rpe_allocation_sensitivity_validation.md`](rpe_allocation_sensitivity_validation.md)
- [`rpe_four_round_accounting_validation.md`](rpe_four_round_accounting_validation.md)
- [`rpe_four_round_phase_validation.md`](rpe_four_round_phase_validation.md)
- [`rpe_target_round_horizon_validation.md`](rpe_target_round_horizon_validation.md)
- [`rpe_delta_round_schedule_validation.md`](rpe_delta_round_schedule_validation.md)
- [`rpe_delta_compiled_cost_validation.md`](rpe_delta_compiled_cost_validation.md)

## 実行・運用

- [`server_parallel_validation_execution.md`](server_parallel_validation_execution.md)：共有CPU/GPUサーバー向けのbounded実行、checkpoint、resume、dry-run
- [`examples/parallel_validation_h4_q1_manifest.json`](examples/parallel_validation_h4_q1_manifest.json)：H4 q=1のdry-run用manifest例

## 発表資料と参考文献

- [`presentations/README.md`](presentations/README.md)：発表資料・構成案の位置づけ
- [`references/README.md`](references/README.md)：同梱した論文PDFの位置づけ
- [`rte_source_versions.md`](rte_source_versions.md)：RTE一次資料の版管理

## 状態の読み方

個別文書に数値があっても、それだけで現在利用可能とは判断しない。
再現可能性、失効、成果物の有無は[`../VALIDATION_STATUS.md`](../VALIDATION_STATUS.md)と
[`../artifacts/validation_manifest.json`](../artifacts/validation_manifest.json)で確認する。

## mainに収録した過去の成果

[収録済み成果と原SOURCE](research/main_collection_20261103.md)：第1回はTrack B G6の既存結果・文書・再現用SOURCE。元の科学的区分とSTOPを保持する。

[枝ごとの収録範囲](research/main_collection_20261103_branches.tsv)と[大きな原データの既存所在](research/main_collection_20261103_raw_locations.tsv)から、main内の成果と原snapshotを辿れる。

## main集約の版差と原SOURCE（2026-11-03）

[索引の統合・SOURCEの版差・保存原版](research/main_collection_versions_20261103.md)から、同名ファイルの各公開版とmainの配置を確認できる。
過去のSOURCE・test・結果は元snapshotの条件で読む。科学実行と元の研究testの再実行は行っていない。

## P1・研究C SOURCEの組（2026-11-03）

[選定したSOURCEと依存版・検査範囲](integration/main_p1_c_price_cohorts_20261103.md)を参照する。P1は人工・mock検査79件PASS、研究Cはpassive importと静的互換を確認した。
共有coreの変更前版は保存済み。過去の科学SOURCE・入力・結果・契約・STOPは元snapshotを使う。

## サーバー照合からの成果収録（2026-11-03）

[収録ファイル・保存版と確認範囲](research/main_server_results_collection_20261103.md)から、過去の公開成果と元SOURCEを辿る。
既存mainの科学ファイルは上書きせず、研究test関連の原版は実行contextと分けて保存する。新科学計算は行っていない。

## ローカルの未コミット資料からの追加収録（2026-11-03）

[原稿・レビュー・研究ノートと保存版の対応](research/main_local_results_collection_20261103.md)。原文の科学的採用や新規実行の認定ではない。
