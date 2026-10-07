> 2026-10-07 Track B RA-D0 v3：**READY_FOR_RA_D0_ONE_SHOT_AUTHORIZATION**（実行承認ではない）。
> [source review](tracks/algorithm_codesign/ra_d0_source_review_v3_20261007.md) / [GPT handoff](tracks/algorithm_codesign/ra_d0_gpt_handoff_v3_20261007.md) / [manifest](../artifacts/track_b_ra_d0_source_review_v3/2026-10-07/evidence_manifest_v3.json)。
> [focused verifier](../scripts/tracks/algorithm_codesign/verify_ra_d0_source_review_v3.py) / [v3 tests](../tests/tracks/algorithm_codesign/test_ra_d0_source_review_v3.py)。
> exact-certified B2 minimum infeasibilityを正常outcomeに修正。pointをfreezeへ記録しbudget/queryを空にして次nへ進む。
> uncertified failureはtechnical STOP。数値・candidate・grid・call/resource capは維持。登録最適化0、authorizationなし、mandatory STOP。

> 2026-10-07 Track B RA-D0 v2：**READY_FOR_SEPARATE_RA_D0_ONE_SHOT_REVIEW**。
> [source review](tracks/algorithm_codesign/ra_d0_source_review_v2_20261007.md) / [GPT handoff](tracks/algorithm_codesign/ra_d0_gpt_handoff_20261007.md) / [evidence manifest](../artifacts/track_b_ra_d0_source_review_v2/2026-10-07/evidence_manifest_v2.json)。
> [future runner](../scripts/tracks/algorithm_codesign/run_ra_d0_one_shot.py) / [focused verifier](../scripts/tracks/algorithm_codesign/verify_ra_d0_source_review_v2.py) / [v2 tests](../tests/tracks/algorithm_codesign/test_ra_d0_source_review_v2.py)。
> B0_saved/ideal分離、数値B1⊂B2⊂B3、profile-paired budget、batch freeze-before-B3、
> anchor-first、main LP 55,275 / auxiliary込み110,550、resource/launch guardを固定。
> 登録最適化・実budget/minimum/witness取得0、authorizationなし。旧本文・旧STOP・Track Aは保持。mandatory STOP。

## Track B RA-D0 static preparation / mandatory STOP（2026-10-06）

[source review](tracks/algorithm_codesign/ra_d0_source_preparation_review_v1.md)：21 columns/x、18 sign pairs一致、35 focused tests PASS。
B専用namespace `src/trottertracks/algorithm_codesign/ra_d0/`、static generator／verifier、
`artifacts/track_b_ra_d0_preparation/2026-10-06/`に候補・grid・semantic audit・provenanceを保存。
ideal nestingと数値membershipを区別し、query実行scope／資源上限をGPTへ返す。
登録最適化／新合成／新science0、RUN_READY=false、既存分類・Track A・旧STOP保持。
**mandatory STOP。以下の既存本文を全文保持する。**

## Track B RA-RTE統合数学監査・mandatory STOP（2026-10-06）

R1.5 `af3d014d0a0cfcbbd25bb544f6544652fec92942` 基点、GPT設計案へのDOCS_SYMBOLIC_ONLY_MATHEMATICAL_AUDIT。
[命題別監査](tracks/algorithm_codesign/ra_rte_mathematical_audit_v1.md)と[GPT handoff](tracks/algorithm_codesign/ra_rte_mathematical_audit_gpt_handoff_20261006.md)：一block／finite table／canonicalのfixed-n LPは仮定付きで成立。
shot-gridの固定total cap保存には反例。log/root・sampler認証、Delta=0、peak workspaceの規約を実行前修正へ返す。
[stdlib人工bookkeeping](../scripts/tracks/algorithm_codesign/check_ra_rte_mathematical_bookkeeping.py)の[50 checks](../artifacts/track_b_ra_rte_mathematical_audit/2026-10-06/bookkeeping_checks_v1.json)を一般証明と分離した。
[manifest](../artifacts/track_b_ra_rte_mathematical_audit/2026-10-06/evidence_manifest_v1.json)、[dated note](research/研究ノート/2026-10-06_track_b_ra_rte_mathematical_audit.md)。science/synthesis/solver/資源再採点0、共通API変更0。
既存科学分類・結果・STOPは不変。性能・新規性・algorithm採択・次実装／R2 authorizationは未確定。
**mandatory STOP。次の採択・実装・pilotの必要性／範囲はGPT判断。以下の既存本文を全文保持する。**

## Track B R1.5保存値帰属・mandatory STOP（2026-10-06）

input R1 commit `24bfeb84a4ce87b56985d174dd98d1d5e1702a2b` の保存値だけを用いたPOSTHOC attribution / design input。
[帰属報告](tracks/algorithm_codesign/r1p5_saved_value_attribution_v1.md)と[GPT handoff](tracks/algorithm_codesign/r1p5_gpt_handoff_20261006.md)：新science/synthesis/compile/候補追加0、R1科学分類は不変。
primaryは2-qubit finite P₃、distinct-basis controlled、x={1/8,1/4}、登録native三precision。
Aは登録(G_T,G_CX,G_1Q) frontにx=1/8の1e-4、x=1/4の1e-3/1e-4で残る。
normalizationだけでなくnative費用とbias/shotの関係を整理し、固定合成列への依存も保存した。
[stdlib保存値解析](../scripts/tracks/algorithm_codesign/analyze_r1p5_saved_attribution.py)、[全summary](../artifacts/track_b_r1p5_saved_attribution/2026-10-06/attribution_summary_v1.json)、[provenance manifest](../artifacts/track_b_r1p5_saved_attribution/2026-10-06/evidence_manifest_v1.json)、
[日付note](research/研究ノート/2026-10-06_track_b_r1p5_saved_attribution.md)。共通library変更・独立validation・新algorithm採択はない。
限定診断SUPPORTS_RA_RTE_DESIGNは設計入力のみ。eta探索/R2/DF接続/追加scienceは未認可。
**mandatory STOP。次の数学設計・研究方針判断はGPT側。以下の既存本文を全文保持する。**

## Track B R1一回結果・mandatory STOP（2026-10-06）

固定S `d43d64a821a0249a0dfab12a2472bd3a72fdee74` →直接子authorization-only A
`411f08f768244fe87b600d82308c3851847fe9e4`からrun1/retry0。
[結果照合](tracks/algorithm_codesign/r1_one_shot_result_validation_20261006.md)：126 keys / 264 rows / 132 controlled tasks完了、全task適格。
terminal R1_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW、[保存field専用監査](../scripts/tracks/algorithm_codesign/audit_r1_saved_result.py) PASS。
primary distinct-basis controlledではB²改善とnative/shot資源のtrade-offを保存し、自動研究GOはない。
[全264 rows CSV](../artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/resource_rows_display_v1.csv)、[evidence manifest](../artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/evidence_manifest_v1.json)、
[GPT判断への入口](tracks/algorithm_codesign/r1_post_run_gpt_review_request_20261006.md)。原result/marker/source、既存証拠・共通API・Track A保持。
science終了後mandatory STOP、追加合成/target/grid/分子/DF/trajectory/GPUは行わない。
研究方針・RQ・新規性・着地点・追加検証の必要性/範囲はGPT側。
以下のpending/最新記述は当時の履歴として本文をそのまま保持する。

# 文書索引

## Track B R1 v2 source review — 2026-10-06

[Preregistration](tracks/algorithm_codesign/rte_reallocation_r1_preregistration_v2.md)、
[native semantics](tracks/algorithm_codesign/rte_reallocation_r1_native_semantics_v1.md)、
[GPT source review request](tracks/algorithm_codesign/rte_reallocation_r1_source_review_request_20261006.md)。
27 focused tests PASS、static126 keys、実synthesis/resource取得0。R1 science未認可、mandatory STOP。


## Track B R0.5 audit — 2026-10-06

[Equivalence/novelty audit](tracks/algorithm_codesign/rte_reallocation_r05_equivalence_novelty_audit_v1.md)、
[GPT handoff](tracks/algorithm_codesign/rte_reallocation_r05_gpt_handoff_20261006.md)、
[dated note](research/研究ノート/2026-10-06_track_b_rte_reallocation_r05.md)。
限定 `METHOD_DELTA_CANDIDATE` / `CONDITIONAL-R1`、科学実行なし、R1未認可、mandatory STOP。


## Track B R0 technical review (2026-10-06)

[Finite-mean reallocation review packet](tracks/algorithm_codesign/rte_reallocation_r0_review_packet_20261006.md) links the independent proof, primary-text claim audit, one conditional R1 proposal and exact-check provenance. This is symbolic evidence, not an executed science pilot; mandatory STOP.


Track B最新は[BS-0.5設計監査](tracks/algorithm_codesign/bs05_method_target_design_audit_v1.md)。
[ordinary形式仕様](tracks/algorithm_codesign/bs05_ordinary_finite_rte_baseline_v1.md)、
[pilot案v2 amendment](tracks/algorithm_codesign/block_synthesis_pilot_amendment_v2.md)。
candidate独立delta未定義、O/Cを分けてmulti-resource Pareto。実装/science/tests0、mandatory STOP、GPT判断待ち。
以下は固定結果・判断履歴。

Track B最新は[SP-1後のblock合成・数学/実装仕様](tracks/algorithm_codesign/block_synthesis_design_review_20261006.md)。
[claim/対照表](tracks/algorithm_codesign/block_synthesis_claim_and_baseline_matrix_v1.md)、
[小型pilotの未承認案](tracks/algorithm_codesign/block_synthesis_small_pilot_proposal_v1.md)。
GPT reviewをdocs-onlyで具体化。新science/実装/tests0、RUN_READY=false、mandatory STOP。
以下のSP-1以前は固定結果と判断履歴として保持する。

Track B最新は[SP-1一回結果・保存値監査](tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md)。
SP1_RESOURCE_MAP_COMPLETE_AWAITING_REVIEW、48 rows／96 axes完了、保存値audit PASS。
[raw result／marker／evidence manifest](../artifacts/track_b_sp1_wrapper_result/2026-10-06/v1/)。
CのDRのみn=8/16でgain、32でloss、64でshot cap。Cのselective-only material gainは0。
local synthetic mechanism evidence。mandatory STOP、retry0、次stage未認可。
[GPT研究再評価依頼](tracks/algorithm_codesign/sp1_post_run_gpt_review_request_20261006.md)へ戻す。
以下のsource/preparation statusは結果前履歴として保持する。

Track B最新は[SP-1採用済み結果前契約・source最終review](tracks/algorithm_codesign/sp1_wrapper_preregistration_v1.md)。
fusionをrole/mask非依存へ修正し、費用を加法的な保存primitive T数に限定した。
全16 pathの静的監査、B実adapter/runner、59 focused testsを公開。science sweep0、実行未認可。
[source manifest](../artifacts/track_b_sp1_wrapper_source/2026-10-06/source_manifest_v1.json)。
新source固定後にGPTの最終reviewへSTOP。以下は既存判断・提案段階の履歴。

Track B最新は[SP-1結果前契約案・共通会計の技術検証](tracks/algorithm_codesign/sp1_wrapper_preregistration_proposal_v1.md)。
wrapper累積・四placement／一catalogueを具体化。17 focused testsはlocal pass、science実行0、RUN_READY=false。
新規性／DF優位は未確定、fusionとactual adapter等は未完了。別source／authorization前に契約案reviewへSTOP。
以下は既存判断・結果前履歴。

Track B最新は[SP-0.5 one-shot結果・GPT handoff](tracks/algorithm_codesign/sp05_one_shot_result_validation_20261006.md)。
PRIMITIVE_TRADEOFF_EXISTS、23 keys／16 rows完了、strict14／controls2。
[元result・consumed marker・保存値監査](../artifacts/track_b_sp05_economics_result/2026-10-06/v1/)。
primitive実装確認のlocal evidence。retry0・mandatory STOP、wrapper pilot未認可。次の研究判断はGPT側。
以下は結果前／既存判断の履歴。


Track Bの現段階は[SP-0.5 synthesis-economics preregistration／source review](tracks/algorithm_codesign/sp05_synthesis_economics_preregistration_v1.md)。
合成器一つ・catalogue一つ・8 targetを固定し、B専用実装／34 focused testsを準備した。登録target合成0、実行未認可。
[preparation manifest](../artifacts/track_b_sp05_economics_preparation/2026-10-06/preparation_manifest_v1.json)。
16-cell wrapper pilotは先行させず、source reviewと別authorizationへSTOP。以下の「最新」は当時の履歴。


Track B最新は[BM-0.5後review・合成placement設計仕様案](tracks/algorithm_codesign/synthesis_placement_design_review_20261006.md)。
B-F限定closure／B-M現new-method closureを維持し、四placement・full wrapper mean・有限合成／測定costを具体化する。
小さい[設計JSON](../artifacts/track_b_synthesis_placement_design/2026-10-06/design_contract_v1.json)はauthorizationではない。
新利益・新規性は未実証、pilot未実行・未認可。以下の「最新」は当時の記録として保持する。

Track Bの最新は[BM-0.5同値性監査・GPT handoff](tracks/algorithm_codesign/bm05_review_packet_20261005.md)。
固定nestedの三次係数と同policy scoreにmethod deltaなし。BM-1を実行せずSTOPし、applicationの価値はGPT判断。

Track Bの最新は[BF-A後review・BM-0 packet](tracks/algorithm_codesign/bm0_review_packet_20261005.md)。
B-F限定closure、B-Mのnative列／DF情報／小型pilot案を収録。科学実行0、BM-1未認可、review待ちでSTOP。

Track Bの入口は[Algorithm Co-design](tracks/algorithm_codesign/README.md)。
[BF1-R0復元結果](tracks/algorithm_codesign/bf1_read_only_recovery_result_validation_20261005.md)は事後のprimary BF-A、
F/L finite最小値同一を保存する。science rerunは0、mandatory STOP。
[BF1-R0契約](tracks/algorithm_codesign/bf1_read_only_recovery_contract_v1.md)は保存済みcellだけの一回の復元を認可する。
原BF-1はINCONCLUSIVEのまま、science rerunは未認可。
原science実行は[BF-1一回実行の結果照合](tracks/algorithm_codesign/bf1_one_shot_result_validation_20261005.md)。
JSON保存例外で`INCONCLUSIVE`、mandatory STOP、retryなし。結果前契約とpreparationは履歴として保持する。
別branchの[JSON保存修正・限定検証](tracks/algorithm_codesign/bf1_serialization_repair_review_20261005.md)は
Codex側の技術作業記録。研究方針全体の修正はGPT側で扱う。

このディレクトリには、研究方針の正本、実装規約、検証報告、発表資料の案内が共存する。
研究全体を初めて読む場合は、先に[`../PROJECT_MAP.md`](../PROJECT_MAP.md)と
[`research/研究概要・現状.md`](research/研究概要・現状.md)を読む。

PR-2別系列の最新結果は
[`M2 held-out結果照合`](pr2_matched_accuracy_m2_transfer_result_validation.md)。
固定5構成、196 wrappersの一回実行は`TRANSFER_SUPPORTED`、研究方針全面review待ちで停止している。
developmentの根拠は[`M1-B1結果検証`](pr2_matched_accuracy_m1_b1_result_validation.md)。
旧S0 STOPとS2結果を保持した
S2後reviewでは、[`matched-accuracy先行研究gate`](research/pr2_matched_accuracy_prior_art_gate_v1.md)、
[`M1前resource-map契約`](research/pr2_matched_accuracy_resource_contract_v1.md)、
[`M1実装契約`](research/pr2_matched_accuracy_m1_implementation_contract_v1.md)、
[`M1前最終amendment`](research/pr2_matched_accuracy_m1_preexecution_amendment_v2.md)を固定した。続いて
[`M1-A validation`](pr2_matched_accuracy_m1_a_validation.md)でdevelopment 1.00 Åの210候補を評価した。
64 proxy-frontier候補中52件が16-cell cap外に残り、`SELECTION_LIMITED`でcompile 0のまま停止した。
外部review後、[`M1-B1 bounded compile契約`](research/pr2_matched_accuracy_m1_b1_bounded_compile_contract_v1.md)で
accuracy適格random 194 cell×32 trajectory×2軸とB0/B1 16 cell×2軸、計12,448 wrapperの有限grid、
cache identity、B1後STOPをzero-compute固定した。
実行前外部reviewの修正要求は、[`M1-B1 execution contract amendment v2`](research/pr2_matched_accuracy_m1_b1_execution_contract_amendment_v2.md)で
科学実行sourceをauthorizationより先に固定し、runnerのterminal statusをcompile map完成review待ちまたは
implementation failureだけに限定した。
actual execution source commit `33f436b`、source-bound plan v2、
[`M1-B1 execution authorization v1`](research/pr2_matched_accuracy_m1_b1_execution_authorization_v1.md)を固定して
12,448-wrapper mapを完了した。検証後の判断は`CONTINUE_RESOURCE_STUDY`で、その時点ではheld-out未認可だった。
後続M2だけを別契約・source・authorization・最終reviewと利用者指示に従って実行した。追加96、S3は未実行・未認可。

## 研究方針と現在地

- [`research/研究概要・現状.md`](research/研究概要・現状.md)：最新の短い全体要約
- [`research/prevalidation_catalog_evidence_map.md`](research/prevalidation_catalog_evidence_map.md)：事前検証カタログの実施IDと文書・artifact・testの対応
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
- [`research/finite_rte_phase_amplitude_contract.md`](research/finite_rte_phase_amplitude_contract.md)：FR-0の補正後演算子・平均演算子、位相・信号半径境界、比較契約
- [`research/finite_rte_phase_amplitude_prior_art.md`](research/finite_rte_phase_amplitude_prior_art.md)：finite RTEと近接解析のscoped先行研究監査
- [`research/finite_rte_phase_amplitude_fr1_preregistration.md`](research/finite_rte_phase_amplitude_fr1_preregistration.md)：FR-1の非可換toy入力・gate・停止規則と実行後status
- [`finite_rte_phase_amplitude_validation.md`](finite_rte_phase_amplitude_validation.md)：G0/G1/G3/G4通過、G2不通過、mechanism-only停止結果
- [`fr_revision_fr1a_posthoc.md`](fr_revision_fr1a_posthoc.md)：既存FR-1を正scalarと強いbaselineで再解析し、scalar-only説明となった事後監査
- [`fr_revision_nonuniform.md`](fr_revision_nonuniform.md)：非一様4×4でFR固有の境界改善を確認したが、固定予算の選択差がなくmechanism-onlyで停止したFR-R1b結果
- [`research/fr_revision_scalar_structure_contract.md`](research/fr_revision_scalar_structure_contract.md)：FR-R0の正scalar分離、情報層、強いbaseline、FR-R1事前登録要件
- [`research/fr_revision_fr1a_posthoc_plan.md`](research/fr_revision_fr1a_posthoc_plan.md)：既存FR-1を再分類しない正scalar事後解析計画
- [`research/fr_revision_nonuniform_preregistration.md`](research/fr_revision_nonuniform_preregistration.md)：非一様4×4の固定grid、状態、比較、GO/STOP
- [`research/fr_research_claim_and_manuscript.md`](research/fr_research_claim_and_manuscript.md)：FR-R1b後の中核主張C1/C2、条件付きC3、先行研究監査、証明義務、完成判定
- [`research/pr2_s0_s1_execution_amendment_v3.md`](research/pr2_s0_s1_execution_amendment_v3.md)：S0実行と、S0通過時だけのS1 correctness実行を許可し、S1 summary後のmandatory STOPを固定
- [`research/pr2_codex_validation_policy_d3e1723.md`](research/pr2_codex_validation_policy_d3e1723.md)：Codexが実装・実行してよいS0/S1範囲とS2/S3禁止を定める方針
- [`research/pr2_s0_reproduction_stop_c644925.md`](research/pr2_s0_reproduction_stop_c644925.md)：development hash不一致による`STOP_INPUT_REPRODUCTION_MISMATCH`、S1未実行、証拠hashと再試行条件
- [`research/pr2_s0_external_review_request_c644925.md`](research/pr2_s0_external_review_request_c644925.md)：S0 terminal STOP後に、終了または新しい結果前amendmentの要否をGPTへ確認するレビュー依頼
- [`research/pr2_v4_s2_development_authorization_v5.md`](research/pr2_v4_s2_development_authorization_v5.md)：V4 correctness、development-only S2、S2後mandatory STOPを結果前固定
- [`pr2_v4_s2_development_validation.md`](pr2_v4_s2_development_validation.md)：V4/S2の実行結果、B2/B3 frontier、rank control、方針review判断
- [`pr2_matched_accuracy_m1_a_validation.md`](pr2_matched_accuracy_m1_a_validation.md)：210候補のcompile-free signal/selector、52未選択frontier、`SELECTION_LIMITED`、全compile counter 0を記録するM1-A結果
- [`pr2_matched_accuracy_m1_b1_result_validation.md`](pr2_matched_accuracy_m1_b1_result_validation.md)：12,448 wrapper、全checkpoint/cache再集計、actual B2 rank-3 frontier、旧selector監査、fixed-q=8比較、状態準備感度と`CONTINUE_RESOURCE_STUDY`を記録するM1-B1結果
- [`pr2_matched_accuracy_m2_transfer_result_validation.md`](pr2_matched_accuracy_m2_transfer_result_validation.md)：固定5構成のH4 1.30 Å transfer、196 wrapper照合、`TRANSFER_SUPPORTED`、primary ratio/paired uncertainty、資源・pre/post testsと研究方針reviewへのmandatory STOP
- [`research/pr2_matched_accuracy_prior_art_gate_v1.md`](research/pr2_matched_accuracy_prior_art_gate_v1.md)：M1前のclaim-level先行研究比較と`PROCEED_RESOURCE_STUDY`判定
- [`research/pr2_matched_accuracy_resource_contract_v1.md`](research/pr2_matched_accuracy_resource_contract_v1.md)：matched-accuracy baseline、可変q correctness、compile選抜、held-out前freezeを定める研究契約
- [`research/pr2_matched_accuracy_m1_implementation_contract_v1.md`](research/pr2_matched_accuracy_m1_implementation_contract_v1.md)：M1の候補identity、seed、selector、schema、zero-compute guardと科学計算未承認を固定する実装契約
- [`research/pr2_matched_accuracy_m1_preexecution_amendment_v2.md`](research/pr2_matched_accuracy_m1_preexecution_amendment_v2.md)：2026年の近接研究二件との最終claim照合と、M1-A limited時にcompile job 0で停止するhard barrierを追加する現行amendment
- [`research/pr2_matched_accuracy_m1_execution_authorization_v1.md`](research/pr2_matched_accuracy_m1_execution_authorization_v1.md)：development-only M1-Aの入力、source、最大212 signal、compile 0、held-out access 0を結果前固定する実行承認
- [`research/pr2_matched_accuracy_m1_execution_authorization_v1_1.md`](research/pr2_matched_accuracy_m1_execution_authorization_v1_1.md)：v1のresult未作成停止後、固定KのRTEConfig self-consistencyだけを修正して同じM1-Aを再認可
- [`research/pr2_m1_a_selection_limited_external_review_request_3c1831e.md`](research/pr2_m1_a_selection_limited_external_review_request_3c1831e.md)：commit `3c1831e`のM1-A結果を固定し、compile上限拡張・technical note・停止の三択をGPTへ依頼するレビュー文
- [`research/pr2_matched_accuracy_m1_b1_bounded_compile_contract_v1.md`](research/pr2_matched_accuracy_m1_b1_bounded_compile_contract_v1.md)：194 random＋16 baseline cell、12,448 wrapper上限、cache/checkpoint identity、B1後STOPを固定し、科学実行を未承認に保つ契約
- [`research/pr2_m1_b1_preexecution_external_review_request_1228168.md`](research/pr2_m1_b1_preexecution_external_review_request_1228168.md)：source commitとzero-compute planを固定し、result-prior M1-B1 authorization作成前のGPTレビュー項目と回答形式を定める依頼文
- [`research/pr2_matched_accuracy_m1_b1_execution_contract_amendment_v2.md`](research/pr2_matched_accuracy_m1_b1_execution_contract_amendment_v2.md)：execution source先行固定と、compile map完成後に研究判断を外部reviewへ戻すterminal status修正
- [`research/pr2_matched_accuracy_m1_b1_execution_authorization_v1.md`](research/pr2_matched_accuracy_m1_b1_execution_authorization_v1.md)：source commit `33f436b`、plan v2、12,448 wrapper、6 workers、2 terminal status、held-out禁止を結果前固定する一回限りの実行認可
- [`research/pr2_m1_b1_execution_authorization_external_review_request_8fc2400.md`](research/pr2_m1_b1_execution_authorization_external_review_request_8fc2400.md)：authorization bundle commit `8fc2400`を固定し、本計算開始前の最終GPT reviewと三択回答形式を指定する依頼文
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
- [`research_direction_prevalidation.md`](research_direction_prevalidation.md)
- [`research_direction_ablation.md`](research_direction_ablation.md)
- [`research_direction_pf_sensitivity.md`](research_direction_pf_sensitivity.md)
- [`research_direction_gate_s1.md`](research_direction_gate_s1.md)
- [`research_direction_structure_pilot.md`](research_direction_structure_pilot.md)
- [`research_direction_sequence_policy.md`](research_direction_sequence_policy.md)
- [`research_direction_full_scope.md`](research_direction_full_scope.md)
- [`research_direction_full_scope_extension.md`](research_direction_full_scope_extension.md)
- [`research_direction_decision_cost.md`](research_direction_decision_cost.md)
- [`research_direction_late_round_proxy.md`](research_direction_late_round_proxy.md)
- [`research_direction_compiler_transfer.md`](research_direction_compiler_transfer.md)
- [`research_direction_uncertainty_break_even.md`](research_direction_uncertainty_break_even.md)
- [`research_direction_wp11_synthesis.md`](research_direction_wp11_synthesis.md)
- [`research_direction_full_opt2.md`](research_direction_full_opt2.md)：WP11選択M06-Fの事前固定条件、51/51完全性監査、direct-RZ測定、coherent opt2再最適化、A0 proxy-lineage再照合
- [`research_direction_signal_weight_pilot.md`](research_direction_signal_weight_pilot.md)：P-Bのenergy bias・target weight・q別signal再解析とテーマ選定判断
- [`research_direction_geometry_energy_difference_pilot.md`](research_direction_geometry_energy_difference_pilot.md)：P-Cのgeometry依存signed PF error、未使用geometry/delta、差分bias予測
- [`research/pc_geometry_tracking_breakdown_preregistration.md`](research/pc_geometry_tracking_breakdown_preregistration.md)：P-Cの8 geometry、追跡規則、blind region、7 gate、停止規則を計算前に固定
- [`research_direction_geometry_tracking_breakdown.md`](research_direction_geometry_tracking_breakdown.md)：追跡prefix不変、stretch予測破れ、固定gateによるcurrent H4 P-C停止
- [`research/pd_energy_tail_pareto_preregistration.md`](research/pd_energy_tail_pareto_preregistration.md)：P-Dの固定5公式、development/blind、7 gate、停止規則
- [`research_direction_energy_tail_pareto.md`](research_direction_energy_tail_pareto.md)：energy-onlyとtail-aware選択のblind逆転、P-D条件付き候補、次のsigned-time/internal-H_D gate
- [`research/pd_realization_go_no_go_preregistration.md`](research/pd_realization_go_no_go_preregistration.md)：P-D現実化の負時間finite-RTE、fragment内部`H_D`誤差、fresh `L_D=5`、Go/No-Go停止規則
- [`research_direction_pd_realization.md`](research_direction_pd_realization.md)：D1--D3全通過、P-D正式候補化と研究再設計停止点
- [`research/pd_primary_research_contract.md`](research/pd_primary_research_contract.md)：P-D S0の主RQ、比較契約、Case A--D、強制停止
- [`research/pd_prior_art_and_baselines.md`](research/pd_prior_art_and_baselines.md)：既知absolute-tail-time modelと新規性候補の境界
- [`research/pd_s1_fair_comparison_preregistration.md`](research/pd_s1_fair_comparison_preregistration.md)：固定時間・位相予算、B0/B1a/B1b/B2/B4、K4・境界規則
- [`research_direction_pd_fair_comparison.md`](research_direction_pd_fair_comparison.md)：B1b/B2/B4一致、Case C/D不成立、B1a境界未解消のS1停止結果
- [`research/pd_s1_posthoc_reanalysis_plan.md`](research/pd_s1_posthoc_reanalysis_plan.md)：固定S1 artifactだけを使う事後再解析の入力、5%近傍、解釈規則、停止条件
- [`research_direction_pd_s1_posthoc.md`](research_direction_pd_s1_posthoc.md)：一次Case Bを保存した主baseline再解釈、B1a診断、nested/native内訳
- [`research/r3_prior_art_and_minimal_contract.md`](research/r3_prior_art_and_minimal_contract.md)：広いR3の重複、R3-S0不通過、実行しない条件付き最小検証契約
- [`../pd_s1_review_5c331f0.md`](../pd_s1_review_5c331f0.md)：S1 snapshotに対する外部レビュー。正式方針ではなく事後再解析の入力資料
- [`research_direction_joint_synthesis_pilot.md`](research_direction_joint_synthesis_pilot.md)：P-Aのinterval-aware DF回路列合成、強いbaseline、未使用列holdout
- [`research_direction_theme_selection.md`](research_direction_theme_selection.md)：P-B/P-C/P-A比較と後続停止点を含む選定履歴。現行判断はA/B/Cに確認済み主題なし
- [`research/pa_joint_synthesis_prior_art_audit.md`](research/pa_joint_synthesis_prior_art_audit.md)：P-A v1のscoped prior-art audit、限定novelty statement、v2境界
- [`research/pa_joint_synthesis_blind_validation_preregistration.md`](research/pa_joint_synthesis_blind_validation_preregistration.md)：H5 physical transferとH4 opt2 compiler transferの事前登録、compile前task manifest、固定gate
- [`research_direction_joint_synthesis_blind_validation.md`](research_direction_joint_synthesis_blind_validation.md)：P-A v1のH5 physical transferとH4 opt2 compiler transferの完了結果、固定gate、判断、scope
- [`research/pa_joint_synthesis_v1_formalization.md`](research/pa_joint_synthesis_v1_formalization.md)：P-A v1の有限候補、DP最適性・計算量・同値性条件、一区間退化と次の機構識別
- [`research/pa_joint_synthesis_mechanism_validation_preregistration.md`](research/pa_joint_synthesis_mechanism_validation_preregistration.md)：明示的一区間baseline、forced support変化、order 2 stream、固定gate・停止規則の事前登録
- [`research_direction_joint_synthesis_mechanism_validation.md`](research_direction_joint_synthesis_mechanism_validation.md)：P-A非退化mechanism検証の0 split・0 plan差・0 RZ改善とP-C復帰判断

## 実行・運用

- [`server_parallel_validation_execution.md`](server_parallel_validation_execution.md)：共有CPU/GPUサーバー向けのbounded実行、checkpoint、resume、dry-run
- [`pr2_s2_parallel_execution.md`](pr2_s2_parallel_execution.md)：固定PR-2 S2のcell-level CPU並列化、段階barrier、persistent compile cache、serial同値性test
- [`examples/parallel_validation_h4_q1_manifest.json`](examples/parallel_validation_h4_q1_manifest.json)：H4 q=1のdry-run用manifest例

## 発表資料と参考文献

- [`presentations/README.md`](presentations/README.md)：発表資料・構成案の位置づけ
- [`references/README.md`](references/README.md)：同梱した論文PDFの位置づけ
- [`rte_source_versions.md`](rte_source_versions.md)：RTE一次資料の版管理

## 状態の読み方

個別文書に数値があっても、それだけで現在利用可能とは判断しない。
再現可能性、失効、成果物の有無は[`../VALIDATION_STATUS.md`](../VALIDATION_STATUS.md)と
[`../artifacts/validation_manifest.json`](../artifacts/validation_manifest.json)で確認する。

## M2 usable B2契約修正 v2（2026-10-04）

外部reviewの修正要求を[amendment v2](research/pr2_matched_accuracy_m2_transfer_contract_amendment_v2.md)へ反映した。
Pareto supportとprimary ratioは共にaccuracy-eligibleかつprimary重大underestimateのないB2だけを使う。
v1証拠・固定5構成・seed・196-wrapper上限を維持し、科学実行とheld-out accessは未認可である。
moduleは`src/trotterlib/pr2_matched_accuracy_m2_transfer_contract.py`、runnerは
`scripts/run_pr2_matched_accuracy_m2_transfer_contract.py`、testは
`tests/test_pr2_matched_accuracy_m2_transfer_contract.py`、schema/planは
`artifacts/pr2_matched_accuracy_m2_transfer_contract/2026-10-04/`から辿れる。

## PR-2 M2科学実行コードの入口

[実装資料](research/pr2_matched_accuracy_m2_transfer_execution_implementation.md)に、固定5構成・196-wrapper
上限、usable B2、paired-axis covariance、one-shot停止、source-bound planと別authorizationを記録した。
sourceとsynthetic検証を固定する段階であり、held-out開封・科学実行・次段階は未認可である。

## PR-2 M2最終実行前レビュー

- [実行authorization](research/pr2_matched_accuracy_m2_transfer_execution_authorization_v1.md)：actual source/plan、固定5構成、196 wrappers、最大5 workers、一回限りを固定する。
- [最終review依頼](research/pr2_m2_execution_authorization_external_review_request_90a9f24.md)：最終承認と利用者の実行指示までheld-out未開封・本計算未実行で停止する。
- `artifacts/pr2_matched_accuracy_m2_execution/2026-10-04/authorization_audit_v1.json`：local zero-science gateとtimed tests。科学結果・immutable CIではない。
