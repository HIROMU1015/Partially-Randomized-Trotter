# 文書索引

## 2026-10-06 H4 signal/compile INPUT_BOUND草案・容量不足STOP

run02の6凍結入力へplanを結合し、source19/218 templates/compiler/run ID/outputを不変に保った。
stage必要3.5 GiB/560000 inodesに対しavailable3.419376 GiB、約82.6 MiB不足。CPU候補6 core/memory/quota観測はPASS。
全72h監視・74784 records・149569 ledger deltas・全8KiB worker logs・1308 signal files・journal/temp/directory余裕を含む。
CPU [3,5,6,7,8,9]・6 worker・own-run mask0x3e8は次段の提案、review=false、利用者の別stage承認/明示launch未取得。
signal/seed/sampling/build/compile/transpile/taskset/worker/GPU/共有環境・他job変更0、入力再生成0。
[次段scope・容量](research/track_a_h4_signal_compile_plan_review.md)と[資料入口・承認対象](../artifacts/resource_applicability/track_a_h4_signal_compile_plan_review/2026-10-06/README.md)を参照する。
`H4_SIGNAL_COMPILE_PLAN_PREPARED_STORAGE_BLOCKED_STOP`。容量と別認可成立後もfresh検査不合格なら起動せず、map後MAP_COMPLETE_STOP。
scientific runtimeはcommitしない。旧10GiB案/旧source/旧bundle/入力生成完了と以下の履歴は保持する。

## 2026-10-06 H4入力生成run02・6入力freeze完了STOP

利用者の明示再実行指示でworker bootstrapのstdlib signal shadowを-Pで修正し、run01を保存してrun02を別固定した。
CPU [3,5,6,7,8,9]・6 workers・own-run限定、fresh resource/容量/quota検査PASS。
H4 linear/STO-3G/DF12の追加6距離入力を一度生成しfreeze完了。NPZ6 bytes SHAを照合し、own driver/worker残存0。
sourceは049e69919af16ad29a67a217dc7a407d6b1754a6。科学/seed/compiler/resource条件は不変、source19変更はgates/worker起動だけ。
`INPUTS_FROZEN_STOP`、next_stage_authorized=false、mandatory_stop=true。signal/compile/GPU/追加transpile0、共有環境・他job変更0。
[修正・完了scope](research/track_a_h4_worker_bootstrap_run02.md)と[完了報告](../artifacts/resource_applicability/track_a_h4_worker_bootstrap_run02/2026-10-06/COMPLETION_REPORT_v1.md)を参照する。scientific runtimeはcommitしない。以下は各時点の履歴。

## 2026-10-06 H4入力生成stage容量確認・最終承認待ち

容量準備の判定は「足りる」。6入力生成→freeze→STOPの必要量3GiB/260000 inodesに対し、
2026-10-06 16:58:53 JSTのnonroot available約3.615GiB、225817022 inodes、user/group/project quota非有効をread-only確認。
32保存配列/NPY・ZIP overhead/64MiB IPC上限/temp-final/72h監視259202 files/journal/metadata余裕を含む。
全campaign10GiBはcharge capとして維持し、旧全量空き確保案を履歴に保存した上でstage-specific補足を追加した。
source19/plan/auth/approved=false reviewはbyte-identical、CPU [3,5,6,7,8,9]・6 worker・own-run mask0x3e8は未承認。
CPU使用許可/独立最終review/明示launch/fresh CPU・memory・pressure/OOM・容量検査が残る。signal/compile容量は別認可。
[容量根拠](research/track_a_h4_input_generation_stage_storage_review.md)と[最終承認資料入口](../artifacts/resource_applicability/track_a_h4_input_generation_storage_review/2026-10-06/README.md)を参照する。
`H4_INPUT_GENERATION_STAGE_STORAGE_CONFIRMED_AWAITING_FINAL_APPROVAL`で公開後STOP。
新科学/追加transpile/taskset/worker/GPU/共有環境・他job変更0。旧10GiB案を含む以下は当時の履歴。

## 2026-10-06 H4入力生成 CPU/launch最終案・利用者承認未取得

提案CPU[3,5,6,7,8,9]、6 worker、異なる6 physical core・NUMA0。約3秒の受動負荷sampleで各core busy0%。
source19 pathsとplan v2 bytes/source_rootは不変。authは候補CPU集合だけ、reviewはauth digestだけ変更しapproved=falseを保持。
own新規runだけにtaskset maskを指定する未実行commandを用意した。CPU許可/専有予約/独立review/明示launchは未取得。
memory/context read-only確認は成功。filesystem空き約3.717GiBは総上限10GiB全量確保案に未達で、launch容量条件は未解決。
12 metadata gate tests PASS、fail/error/skip0。観測01/02の失敗logと03の容量未解決記録を保持し、追加transpile0/旧28件不変。
[提案資料](research/track_a_h4_input_generation_cpu_launch_proposal.md)と[bundle・一括承認判断](../artifacts/resource_applicability/track_a_h4_geometry_input_generation_cpu_launch_proposal/2026-10-06/README.md)を参照する。
`H4_INPUT_GENERATION_CPU_LAUNCH_PROPOSAL_FROZEN_AWAITING_APPROVAL`で公開後STOP。taskset/科学/worker/GPU/共有環境・他job変更0。

## 2026-10-06 H4入力生成 resource observer修正・未承認草案再固定

真のv2 hierarchy rootをnamespace/mount/所属から判定し、rootの非root memory interface要求を修正した。
全可視非root祖先の制限・pressure/OOMは保持し、非root欠測/不明namespace/hidden mountはSTOP、host-only fallbackなし。
observer33 zero-science tests PASS、実read-only観測成功。準備観測available約981.826GiB、PSI/OOM0はlaunch成立ではない。
production変更はresources observerとgateのnew audit pathだけ。科学/並列/seed/compiler source15件は不変。
[実装資料](research/track_a_h4_geometry_resource_observer_fix.md)と[source bundle](../artifacts/resource_applicability/track_a_h4_geometry_resource_observer_fix/2026-10-06/README.md)、
[new認可草案v2](../artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06-v2/README.md)を参照する。source固定→別草案commit、binding検査結果はv2へ記録。
requested workers6、allowed_cpus=[]、approved=false。CPU/launch context・独立最終review・明示launch未解決、実行準備完了とはしない。
`H4_INPUT_GENERATION_RESOURCE_FIX_FROZEN_AWAITING_REVIEW`で公開後STOP。科学/追加transpile/GPU/本番起動/共有環境・他job変更0。
旧bundle/科学証拠/原稿/Track Bと旧監査履歴を保存し、系列transpile28/64を維持する。

## 2026-10-06 H4 geometry 入力生成専用認可草案・実行未承認

凍結science source6a121725（17 Python＋親2件）を変えず、入力生成source-bound planとresult-prior認可草案を追加した。
requested workers6、inputs/freeze digestはnull、218 templatesを機械転記。reviewはapproved=false、allowed_cpus=[]。
CPU許可は利用者指示で未確定のまま。現在process CPU0–255を許可とみなさず、launch contextとmemory観測は未解決。
既存observerはroot cgroup memory.max欠落で停止し、実行準備完了とはしない。source/共有設定を緩和しない。
新57 zero-science gate tests PASS、fail/error/skip0。合格経路はメモリ内模擬承認だけ、追加transpile0・旧累積28/64不変。
[実装・停止条件](research/track_a_h4_geometry_input_generation_authorization_draft.md)と[bundle・最終レビュー入口](../artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06/README.md)を参照する。
`H4_INPUT_GENERATION_AUTHORIZATION_DRAFT_FROZEN_AWAITING_REVIEW`で公開後STOP。科学/GPU/本番起動/共有環境・他job変更0。
有効execution authorization0、final review/利用者の明示launch未実施。入力生成・本計算・signal/compile認可・H6/Track Bへ進まない。

## 2026-10-06 H4 geometry compile並列source再固定・科学未実行

compileの逐次waitをadmitted worker数以下のbounded投入・回収へ変更した。処理中ownerを追跡し、
COMPLETEとidentity/digest検査後だけ再利用する。trajectory/axis順・weight、科学scope・seed/compilerは不変。
[実装資料](research/track_a_h4_geometry_parallel_source_implementation.md)と[new bundle](../artifacts/resource_applicability/track_a_h4_geometry_parallel_source/2026-10-06/README.md)を現在の入口とする。
既存94＋並列回帰17＝111 synthetic tests PASS、fail/error/skip0。今回transpile3、旧25＋新3＝28/64。
fake futures/mock workersだけで制御を検査し、実worker/production性能は未検証。旧bundle/audit/契約・保存証拠は不変。
SOURCEとsourceを変更しないREVIEWの2 commitを分ける。分子アクセス/科学処理/GPU/本番起動/認可発行/共有環境・他job変更0。
`H4_GEOMETRY_PARALLEL_SOURCE_FROZEN_AWAITING_REVIEW`で公開後STOP。入力生成plan/auth作成・本計算・H6/Track Bへ進まない。

## 2026-10-06 H4 geometry server-native source固定・科学未実行

利用者の新指示で契約v2 D1〜D4を実装条件へ採用し、旧未承認履歴を保存した。
新namespace `src/trottertracks/resource_applicability/h4_geometry/`、二つのfuture runner、専用synthetic testsの入口は
[実装資料](research/track_a_h4_geometry_server_native_source_implementation.md)と
[bundle](../artifacts/resource_applicability/track_a_h4_geometry_source/2026-10-06/README.md)。
最終94 tests pass、fail/error/skip0。失敗・再検査込みsynthetic transpile25/64、旧benchmark128再実行0。
SOURCE_COMMITとsourceを変えないREVIEW_BUNDLE_COMMITを分離し、actual blob/hashは別監査で固定する。
旧247 source・v1/v2 bundle・準備25 files・保存6 JSONは不変。分子入力/科学処理/本番runner launch/GPU/環境・他job変更/認可発行0。
`H4_GEOMETRY_SOURCE_FROZEN_AWAITING_REVIEW`で公開後STOP。別入力生成authorizationの作成へ進めるかをレビューし、今回は発行・実行しない。
以下は各milestone当時の履歴。


## 2026-10-06 H4 geometry契約v2・レビュー待ちSTOP

現在の入口は[契約v2 bundle](../artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06-v2/README.md)。v1 commit `7c1a3d43f61c5501a9e79206b7c60933f94b1077`を保存し、
D1〜D4を具体的な採用案、memory admissionを8+8w+16 GiB、認可を入力生成→freeze STOP→別signal/compile認可へ分離した。
H4 linear/STO-3G/DF rank12、6距離・218 template・32 paired trajectories・74,784上限は不変。8 system＋ancilla1(index8)、合計9 qubits。
[pure JSON validator](../artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06-v2/contract_validator_v2.py)と[専用合成検査](../artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06-v2/run_contract_tests_v2.py)はreview用で、science source/runnerではない。
新規320件pass（fail/skip0）、旧129件は保存・runner再実行0。旧v1 manifestはbase blobで照合し書き換えない。
D1〜D4レビュー承認は未解決、science/source port/input generation/next stage認可false、plan未seal、mandatory STOP。
今回の公開指示は軽量契約bundleと関連文書だけのcommit/non-force push。以下は各段階当時の履歴。

最新のTrack Aは、利用者の新しい意向により原稿作成を保留し、
[H4 geometryと要求精度の全候補map](research/track_a_geometry_precision_extension_proposal_v0.md)を準備する段階。
設計案・予算・未確定条件は[別JSON](research/track_a_geometry_precision_extension_proposal_v0.json)へ分離した。
[GPUサーバー側準備指示](research/gpu_server_track_a_h4_geometry_resource_preparation_prompt.md)は既存server環境を優先し、
環境inventory・synthetic CPU benchmark・契約草案まで。本計算はこのhandoffだけでは起動しない。
距離・実行場所・worker・新science source/authorizationは未固定で、本計算は未実行。
旧原稿・証拠・STOPは保存する。以下は各milestone当時の履歴。

Track Aの現在の入口は[原稿v0.1・Supplement・claim audit・review依頼](manuscripts/README.md)。
固定artifactから4主図と補足図1を生成し、日本語通し原稿を作成した。科学計算0、科学STOPを維持する。
原稿のscope監査は執筆者によるもので、独立投稿可能性reviewは次の段階。以下は各stage当時の履歴。

最新のTrack AはPM-2後reviewを採用した[主張・証拠対応表](research/track_a_post_pm2_claim_evidence_map.md)と
[原稿・主要4図の設計](research/track_a_post_pm2_manuscript_design.md)。
追加計算をせず限定case studyとして原稿化する。現在は設計までで、図生成・通し原稿は未実施。
根拠path・固定commit・証拠階層・先行研究との差分・非claimを整理した。
既存result/status/manifest不変、科学計算STOP、Track B変更0。以下のreview待ち等は各stage当時の履歴である。

最新のTrack Aは[PM-2精度・資源境界結果と照合](pr2_pm2_precision_resource_result_validation.md)。
保存JSONだけの302点・67,346行、元ε再現、paired uncertainty、P envelopeを照合し、研究方針review待ちSTOP。
新しい科学計算0、利用者指示でresult commitへ収録するPOSTHOC local evidence。以下は各milestone当時の履歴として読む。

現在は[PM-2保存値解析実装](research/pr2_pm2_precision_analysis_implementation.md)と
[source固定監査](../artifacts/resource_applicability/pr2_pm2_precision_implementation/2026-10-05/source_freeze_v1.json)の段階。
62 local synthetic tests合格、source `324435d77b6642dbd44e8d1f178420daf62e77ed`、本解析は未実行。
研究契約は不変で、別の明示解析指示後も全statusでmandatory STOPする。

最新のTrack A準備は[PM-2精度と資源境界契約](research/pr2_pm2_precision_resource_contract_v1.md)。
全218 development候補とM2元5構成を別集合で固定し、schema・input identity・成果物・STOP条件まで準備した。
保存JSON/input coverageだけを照合し、精度解析は未実施・未認可。旧結果は変更しない。

Track Aの最新は[PM-1実行結果・照合](pr2_pm1_discard_result_validation.md)。
H4 1.00 Å、STO-3G、DF rank12、T=0.8、B0 rank4/5 × q=1/2/4/8の8件がaccuracy適格、
8 signals/16 wrappersを一回完了。pre/post201 local tests passed。
新B0最小rank5・q1は保存B2 r4のprimary点推定の1.75659倍、旧rank6 discardより9.34%低い。
`PM1_DISCARD_MAP_COMPLETE_AWAITING_REVIEW`、mandatory STOP、研究判断null。
結果と監査は利用者指示によるresult commitへ収録するlocal evidence。旧draft/preparation/finalizationの未実行記述は当時の履歴として保持する。

最新のTrack A claim限定は[PM-0 POSTHOC証拠帰属・機構解析](research/pr2_post_m2_evidence_attribution.md)。
元のM1/M2結果と区別し、同一domain、selector指標別regret、N×cost、same-R、欠測の表を辿る。
後続の[PM-1近接discard契約・実装](research/pr2_pm1_nearby_discard_contract_v1.md)は
8 signal＋16 wrapperのfuture上限とsealed planを準備した段階。本計算・authorizationは未実施。
[GPTへのPM-1準備bundleレビュー依頼](research/pr2_pm1_preparation_external_review_request_fd7552e.md)から
PM-0の根拠、固定source、plan、監査へ辿れる。

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


## H4 geometry 契約準備 v1（2026-10-06・local未commit）

[準備bundle](../artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06/README.md)は契約schema・zero-compute plan・pure JSON validatorと129合成検査の入口。
6距離、218 template/点、74,784 wrapper、最大12 workersを固定し、生成/seed/memory/wall/outputはreview待ち。
science source/runnerの追加ではなく、旧公開draft・source・科学結果は不変。本計算・port・commit/pushは未認可、STOP。


## H4候補間compile投入・12 worker明示再実行

利用者の増員再実行指示により[run03 source・認可・検査記録](research/track_a_h4_cross_candidate_run03.md)を追加した。候補内2回路の完了待ちで4 workerがidleとなる問題を、候補間bounded queueで修正。旧run02はworker failure STOPで全証跡を保持し、旧6入力を再生成せず利用する。49人工job/metadata testsはlocal PASSで科学的結果ではない。12 workerのfresh CPU/memory/容量/hash検査後だけ一度起動しMAP_COMPLETE_STOP。旧累積bytes/wall/actual invocationsを引継ぎ、科学条件・compiler・上限は不変。


## H4 run04：identity hash分割・5秒監視維持

[run04固定sourceと再実行binding](research/track_a_h4_streaming_monitor_run04.md)を追加した。run03はmonitor STOPで旧全証跡を保持。人工JSONで旧一括encoderのmonitor interval/freshnessを再現し、新streaming encoderは同じdigestで全監視PASS。65 local tests PASSは実装証拠で科学結果ではない。旧6入力を再生成せず、bytes/wall/7 consumed invocationsをcarryし、CPU12件・12 workerのfresh検査後に一度実行してMAP_COMPLETE_STOP。


## H4 run05：identity hash分割・5秒監視維持

[run05固定sourceと再実行binding](research/track_a_h4_lazy_identity_run05.md)を追加した。run04はmonitor STOPで旧全証跡を保持。人工JSONで旧一括encoderのmonitor interval/freshnessを再現し、新streaming encoderは同じdigestで全監視PASS。75 local tests PASSは実装証拠で科学結果ではない。旧6入力を再生成せず、bytes/wall/12 consumed invocationsをcarryし、CPU12件・12 workerのfresh検査後に一度実行してMAP_COMPLETE_STOP。


## H4 新サーバー引継ぎ（2026-10-07）

[引継ぎ資料入口](research/track_a_h4_new_server_handoff_20261007.md)と[新host Codex指示](research/handoffs/h4-new-server-20261007/NEW_SERVER_CODEX_INSTRUCTIONS.md)。run05は監視STOP、完成compile0/signal0。source19/plan/auth/reviewは不変で、入力・実runtimeはGitに含めない。新hostでは監査・修正・草案固定まで、本計算前STOP。


## 2026-10-07 H4新host取得・監査

[H4新host取得・監査](research/track_a_h4_new_server_preparation_20261007.md)。指定commit/source19照合済み、環境18 version/45 RECORD差の解決案でSTOP。source修正・新tests・production未実施、入力/停止証拠別送待ち、allowed_cpus=[]/approved=false。旧科学結果・原稿・Track Bは保持。


## 2026-10-07 H4新host A案 source/人工検証固定・本計算未認可

[監視修正・32人工tests](research/track_a_h4_new_server_monitor_fix_a_20261007.md)。既存private venv不変で準備用A案採用。
旧source19は3変更/16不変、新module込み25 closure。256×256人工matrix＋9-qubitの旧/new byte/digest一致、独立observerのGIL/GC観測・I/O delay/EOF/所有・資源境界を検証。
SOURCE `b2a5ad89e8b39d72716f7ddb17d263bd0cdedb45`、production/追加transpile0、旧28/64・benchmark128保持。環境18 version差、旧45 raw-reference RECORD差保持・normalized22差を明記。入力6/freeze/runtime/control未受領。
observer AS256MiB/RSS64MiB/admission120.25GiB、容量5.5625GiB案は未承認。allowed_cpus=[]/approved=false/runtime_authorization=false/launch=null、STOP。科学成果/原稿/Track Bと旧資料を保持。


## 2026-10-08 H4本計算前準備・最終承認待ちSTOP

[統合入口・最終承認案](research/track_a_h4_prelaunch_preparation_20261008.md)。新host schema/profile/observer/CPU/one-shot/累積budget bindingを整備。
SOURCE `ad57d1639133f7158cce58d767b8e0aa179bf044`、32 closure、57限定人工tests PASS。core quota read-only確認、worker12＋driver/observer各1別coreを提案。
carry20/165214360 bytes/5466.188392877579秒、残74764を保持。全74784 logicalを保証する最小actual cap+20→74804案は未承認。
候補環境はprivate venv不変、18 version/旧参照対raw45・normalized22差。追加output5GiB/301000 inodes、charge約8.29GiB、copy前は暫定6GiB。
入力6/freeze/native stop proof未受領、allowed_cpus=[]/approved=false/runtime_authorization=false/未seal、science/追加transpile/GPU/共有環境・他job変更0。明示承認・final review・launch前にSTOP。


## 2026-10-08 H4受領監査・独立最終review TECHNICAL FAIL

[統合入口](research/track_a_h4_receipt_final_review_20261008.md)。origin79827016/SOURCEad57d163照合、source32不変、read-only live profile/資源/quotaを確認。
凍結NPZ6/freeze/native stop/controlは未受領、producer bytes/SHA manifestを含む最小転送手順を具体化。科学array読込/再生成0。
独立reviewはpidfd送信時ESRCH競合で後続cleanupが中断するP1を純mock再現しTECHNICAL FAIL。57人工PASSはこの競合を覆わない。
未受領NOT_EVALUABLE、CPU/env/observer/74804案のUNAPPROVEDと技術FAILを区別。sourceは修正せず再sealなし、flags false、carry20/165214360/5466.188392877579を保持し本計算STOP。

## 2026-10-09 H4 cleanup ESRCH修正・独立再review PASS

[新SOURCE・独立再review・残る承認条件](research/track_a_h4_cleanup_esrch_fix_20261009.md)。旧992c09d6から独立worktreeでP1を修正し、SOURCE `6bd1ba01cd71ec3e2071082963c9f07478dada9a`・closure33を固定。
両pidfd送信経路はESRCHだけ既退出扱いで後続cleanupを継続し、他の送信error・所有検証を保持する。
限定人工39件PASS、別担当13純mock件PASSとbinding照合でP1_SCOPE_TECHNICAL_PASS。wait/reap/FD/pipe/first STOP理由保持まで確認。
新test `tests/tracks/resource_applicability/test_h4_cleanup_esrch.py` と既存人工runnerで追跡し、source/profile/plan/auth/reviewを新SHAへ再結合した。
入力0/6・freeze/native停止証拠未受領はNOT_EVALUABLE、未seal。環境/CPU/observer/74804案は未承認、carry20/165214360 bytes/5466.188392877579秒・現actual cap74784不変。
approved=false、runtime_authorization=false、allowed_cpus=[]。追加transpile/科学actual/本番起動0。旧資料を保存して本計算STOP。

## 2026-10-09 H4全byte受領・carry合格・native終端proof待ち

[受領・binding・最終承認案](research/track_a_h4_byte_receipt_binding_20261009.md)。packet86691840B/全2150files/既知9SHAをbyte-only照合しPASS、NPZ6/freeze受領完了。
SOURCE `6bd1ba01cd71ec3e2071082963c9f07478dada9a`・closure33不変。ledger chain/cumulative journalからcarry20/165214360 bytes/5466.188392877579秒を保持、現cap74784・残74764。
run05 log/exact旧sourceからworker cleanup到達は推認可能だが、driver/12 workers停止後identity/残存0 native proofが不足。古いrun02停止監査をrun05proofへ流用しない。
input/profile/source/output/carryとv6 plan/auth/reviewを結合し、control82件のbasename・単一link・sender manifest mappingを照合。source条件は緩めていない。
sealed=false/approved=false/runtime_authorization=false/allowed_cpus=[]。環境/CPU/observer/累積74804案/一度のmap launchは未承認。科学array読込/新科学actual/追加transpile/共有設定変更0でSTOP。

## 2026-10-09 H4追加native停止proof合格・technical再seal

[再seal・独立最終整合review・一括承認案](research/track_a_h4_native_proof_seal_20261009.md)。追加JSON17521B/SHA一致、旧host/run05の13identity・2回残存0・元3証拠hashを照合して現在のnative停止条件PASS。
過去のexit code/正確な終了・reap時刻/原boot IDは未記録のままnull。今回の観測で補完せず、連続監視や歴史cleanup順の証明とも扱わない。
SOURCE `6bd1ba01cd71ec3e2071082963c9f07478dada9a`・closure33不変、profile/input/carry/control83件の固定validator合格でplan再seal、sealed=true。
approved=false/runtime_authorization=false/allowed_cpus=[]。carry20/165214360 bytes/5466.188392877579秒、現cap74784・残74764を保持。
候補environment/compiler・CPU・observer・累積actual74804案・一度のmap launchは未承認。承認による最終artifact/digest再結合とfresh resource/CPU/fs/inode/quota gateをlaunch前に確認。
科学array読込/新science actual/追加transpile/source変更/共有設定変更/本計算0でSTOP。

## 2026-10-09 H4一度のmap実行を利用者承認・最終artifact固定

[実行認可・直前gate・起動報告入口](research/track_a_h4_authorized_launch_20261009.md)。利用者の明示認可で候補environment/compiler採用、worker12 CPUs2/4–6/8–15・driver16・observer18、observerAS256MiB/RSS64MiB/admission120.25GiBを認可。
carry20/165214360 bytes/5466.188392877579秒を保持し、累積actualだけ74804へ+20改定。SOURCE6bd1ba01・science/compiler/options/他caps不変。
sealed/approved/runtime_authorization=true、allowed_cpusはexact14role集合。独立v8 review PASS。artifact commit後fresh CPU/memory/PSI/OOM/FS/inode/quotaとSOURCE/profile/input/carry/unusedrootを確認しPASSなら追加承認なし一度起動。
既存proof/回帰/benchmark再実行0、追加準備campaign/transpile0。oldpartial/cache/GPU/共有環境・venv・他job変更なし、完了またはfail-closed STOP後終了・retry/次stageなし。
これは認可artifact固定時点のsnapshot。実起動/PID/状態は入口への追記・外部runtime receiptで別記録する。

## 2026-10-09 H4 library cache保存先修正・再実行予算不合格

[修正・再実行条件](research/track_a_h4_library_cache_fix_20261009.md)。前回のOpenFermion→Matplotlib mkdir拒否を、homeの新private library cacheへprocess限定MPLCONFIGDIRを結合して修正。
driver/workerとも既存directoryのEEXIST probe以外のcache writeを拒否、29816BのSHA固定。47限定回帰＋3 import case PASS、科学array/transpile/実worker/affinity/GPU0。
SOURCE `b8b3ce6e8c98f1ec0419a7af79c5d7c5f3a3b9bb`、36 closure、science/compiler/options不変。前回費用を返却せずcarry20/4428938712B/5472.345380863175sへ結合。
累積worst charge13165893832B=12.261694GiB>承認10GiBで未seal/approved=false/runtime_authorization=false、再起動0。13GiBは未承認proposalのみ。
既承認environment/CPU/observer/actual74804を保持。cap改定・新SOURCE/gate binding/review・fresh gate後の一度再実行が残る。
private homeは共有systemと区別し、旧run/one-shot/失敗証拠・全予約課金を保持する。

## 2026-10-09 H4軽量高速化・限定同等性確認

実装[signal](../src/trottertracks/resource_applicability/h4_geometry/signal.py)・[ledger](../src/trottertracks/resource_applicability/h4_geometry/ledger.py)、限定[runner](../scripts/resource_applicability/run_h4_lightweight_speedup_tests.py)・[tests](../tests/tracks/resource_applicability/test_h4_lightweight_speedup.py)・[bundle](../artifacts/resource_applicability/track_a_h4_lightweight_speedup/2026-10-09/README.md)。

[変更・限定検証・binding](research/track_a_h4_lightweight_speedup_20261009.md)。driverの距離内共通準備を再利用し、one/DF block呼出を静的13×218→13、全prepareを218→10種類に削減。ledger deltaは変更entry/reservation各最大1だけを参照し、全件走査・saved-historyコピーを除いた。
SOURCE `4d2d1492fc23d0736c305533d78967cc1db8a7c8`、closure38。限定48人工PASS、全218prep/代表8wrapper+dense256case1/代表4signal/ledger13fileの旧new bytes・digest一致。12workersはmock、単一test process内部thread1、science array/transpile/GPU/affinity/production0。
実Gaussian/旧compiler output/実速度・H4本体成功は未検証。monitor/caps/compiler/science/carry不変、旧partial/cacheと混合しない。
carry20/4428938712B/5472.345380863175s、actual74804既承認、worst charge12.261694GiB>承認10GiBは残る。未seal/approved=false/runtime_authorization=false、absolute_launch_command=null、本計算0。

## 2026-10-09 H4利用者が13GiB累積charge・一度の再実行を明示認可

[認可・source・一度の起動入口](research/track_a_h4_approved_relaunch_20261009.md)。利用者の「これについては問題ないので再実行して」を、既存13GiB cumulative charge案と一度のmap再実行の承認として反映。
SOURCE `a7b617600cd7063f7870f2059d5694ef00283f0e`/closure39、output改定schema/gate/実OutputBudget capとmarginのみ変更。legacy10GiB default・科学/compiler/その他caps・prepare再利用/ledger保存は維持。
限定22pure gate PASS、旧48speedup/library/cleanup/native証拠campaign再実行なし。carry20/4428938712B/5472.345380863175s返却なし、actual74804・新残74784。
worst13165893832B <= 新cap13958643712B、余裕792749880B。sealed/approved/runtime_authorization=trueの認可artifactへ再結合。
独立review/artifact固定後fresh SOURCE/profile/input/carry・CPU/memory/PSI/OOM/fs/block/inode/quota/unusedroot/one-shot合格時にそのまま一度起動。実run状態はruntime証跡へ記録。
既承認12workers CPUs2/4–6/8–15、driver16/observer18、thread1・observerAS256MiB/RSS64MiB/admission120.25GiB保持。自動retry/入力再生成/旧partial/cache/次stage/GPU/共有設定変更なし。


## H4 worker error修正（2026-10-09）

[原因保存・独立review・限定検証](research/track_a_h4_worker_error_fix_20261009.md)：39人工PASS、元production例外未特定、新launch未認可。


## H4 run03準備（2026-10-09）

[人工compileを省略・carry結合・追加上限2件](research/track_a_h4_production_run03_20261009.md)。本体未起動。

## 2026-10-09 累積17GiB・74805を明示承認、一度のrun03起動へ

[認可・起動入口](research/track_a_h4_production_run03_20261009.md)。利用者の「上限の変更を承認するので本計算に入って」に基づき、累積output13→17GiBとactual74804→74805を承認済みとして反映。
SOURCE6e68fd9bcc68e788db6f5d43eaa6a03866e53d3b/closure44・科学/compiler/options不変。carry21/8692723164B/5766.582514658794s upper保持、消費/予約返却なし。
人工compile省略、既存32 pure gateと独立技術review/proof/benchmark campaignを再実行しない。environment/CPU/observer・72h/他caps採用済条件を保持。
新plan sealed/auth approved/runtime true・exact14CPUsへ再結合。別担当の最終review、immutable artifact、直前fresh gate/未使用one-shotの合格後、一度mapを起動する。
完了/STOP後終了、自動retry/次stage/入力再生成/旧partial-cache混合/GPU/共有環境・venv・他job変更なし。起動/PID/結果は外部runtime receiptで別記録。

## H4 entrypoint cache修正（2026-10-09）

[原因保存・private memory cache・8限定metadata検証](research/track_a_h4_entrypoint_cache_fix_20261009.md)。別namespaceのlibrary_cache v2 policy、runner run_h4_entrypoint_cache_tests.py。科学compile/追加本計算なし。


## 2026-10-09 H4：失敗分を次回の予算に加算しない

利用者の「失敗した過去分は残さなくてよい」により、次回の新規mapはcarry0（actual/charge/wall）で実行単体を上限の対象とする。
旧carry22/12956511264B/6004.111340102032sと停止証拠・入力は履歴として保持。全map静的見積り8.136939GiB/74784件は承認済み17GiB/74805以内、旧21GiB/74806増額案は不要。
SOURCE cd162e9305143d81c28908b9732f8eef12cd6b89は不変で、現validatorは消費済みrun03/CARRY21。次の新SOURCE/seed/plan/auth/reviewと未使用runをcarry0へ再結合する必要がある。
予算方針のみ承認済み、新one-shot指示/本体起動なし。flagsfalse/allowed[]/未seal/commandnull、科学/追加transpile/GPU/共有環境変更0。
[最新の予算方針・見積り](research/track_a_h4_per_attempt_budget_20261009.md)。


## 2026-10-09 H4 run04：carry0で本計算、STOP後も修正・再実行を継続

利用者が過去失敗分を次回へ加算しない方針と、STOP後チャットを終了せず原因調査・修正・再実行の継続、本体起動を明示認可。
新SOURCE d31b51080665a7eea806ff1adc17f3e0d19151fc/closure49、library cache v2保持、新run04。純粋11 tests PASS、人工compile/追加transpile0。
carry0（actual/charge/wall）・既承認17GiB/74805/72h、全map8.136939GiB/74784。環境/CPU12workers/driver16/observer18/資源caps/科学compiler不変。
旧証拠・入力・one-shotを保持し、各fresh attemptを独立認可binding。停止した同一runのresumeと旧partial科学cache混合は禁止。
独立review・immutable artifact・直前fresh gatesを経て本計算、一度のSTOPで作業を終了せず必要修正とfresh再起動を続ける。次stage未認可。
[最新入口](research/track_a_h4_production_run04_20261009.md)。


## 2026-10-09 H4 run05：carry0で本計算、STOP後も修正・再実行を継続

利用者が過去失敗分を次回へ加算しない方針と、STOP後チャットを終了せず原因調査・修正・再実行の継続、本体起動を明示認可。
新SOURCE 9d1471aff5840a76fa1579f4e14e71e62b8497a3/closure52、library cache v2保持、新run05。純粋16 tests PASS、人工compile/追加transpile0。
carry0（actual/charge/wall）・既承認17GiB/74805/72h、全map8.136939GiB/74784。環境/CPU12workers/driver16/observer18/資源caps/科学compiler不変。
旧証拠・入力・one-shotを保持し、各fresh attemptを独立認可binding。停止した同一runのresumeと旧partial科学cache混合は禁止。
独立review・immutable artifact・直前fresh gatesを経て本計算、一度のSTOPで作業を終了せず必要修正とfresh再起動を続ける。次stage未認可。
[最新入口](research/track_a_h4_production_run05_20261009.md)。


## 2026-10-10 H4 run06：carry0で本計算、STOP後も修正・再実行を継続

利用者が過去失敗分を次回へ加算しない方針と、STOP後チャットを終了せず原因調査・修正・再実行の継続、本体起動を明示認可。
新SOURCE 697843fbbd2a2aa7224261aa26f8da141ca57687/closure55、library cache v2保持、新run06。純粋21 tests PASS、人工compile/追加transpile0。
carry0（actual/charge/wall）・既承認17GiB/74805/72h、全map8.136939GiB/74784。環境/CPU4workers/driver16/observer18/資源caps/科学compiler不変。
旧証拠・入力・one-shotを保持し、各fresh attemptを独立認可binding。停止した同一runのresumeと旧partial科学cache混合は禁止。
独立review・immutable artifact・直前fresh gatesを経て本計算、一度のSTOPで作業を終了せず必要修正とfresh再起動を続ける。次stage未認可。
[最新入口](research/track_a_h4_production_run06_20261010.md)。


## 2026-10-10 H4 run06もhost PSI0.18でSTOP、inactive修正案を固定

run06は4workersで約856.585秒、host PSI0.18/nonroot全0/OOM増分0/available約445GiB/role8GiB内でもmemory_pressure STOP。
4actual予約/完了0・signal0、6identity残存0を2回確認。STOPの発生元や正確な終了時刻は補完しない。
元D4 contractのmemory PSI full avg10>0即STOPを変更するには明示契約改定が必要。通常の修正/再実行認可をpressure変更に拡張解釈しない。
SOURCE41d2b61f640937255c5fa6cab9549ae67b83c462/closure58、旧55 source不変、inactive evaluator・16pure tests PASS。
提案はhost-only PSI<1%を完全なnonroot PSI0/OOM0delta/effective120.25GiB/fresh5秒/既存role8GiB guardで条件付き許容。nonroot>0/host>=1%/欠測等はSTOP。
threshold最適性/共有負荷安全性/本計算成功は未検証。production未接続、approved/runtime false・allowed[]・未seal・commandnull。
carry0/17GiB/74805/72h/科学/compiler/凍結入力維持。本体再起動0、GPU/共有環境/venv/他job変更0。
[最新入口](research/track_a_h4_host_pressure_policy_fix_20261010.md)。


## 2026-10-10 H4 run07：host-only PSI例外の明示承認・本番接続

利用者が「この条件を承認して再実行」と明示承認。host-only PSI<1%は全nonroot0/OOM増分0/effective120.25GiB/fresh5秒でのみ許容、他pressure/欠測/8GiB/5秒等はfail-closed維持。
SOURCE1dbdd1be2133a59ab81c7b81a6074af0880f72a3/closure62、33pure tests PASS、初回失敗ログ保持。actual kernel baselineと原driver OOM/hierarchyをobserver子へ固定transport。
run06原STOP/費用/6×2ABSENT/10filesSHAを保持し、新run07/carry0/CPU4subset[2,4,5,6]/driver16/observer18で再実行。17GiB/74805/72h・科学/compiler/凍結6入力不変。
独立review・immutable artifact・直前fresh/未使用lock PASS後に起動。共有環境/venv/他job/GPU変更0。開始確認後chat終了可、observer監視は継続。
[最新入口](research/track_a_h4_production_run07_20261010.md)。


## 2026-10-10 H4 run07 STOP原因監査：q2 workerのAS8GiB枯渇が最有力

run07はgeometry0.70/B0/L_D3/q1/delta0.8の2wrapperをcompile COMPLETEしsignal1保存後、q2/delta0.4/cosine workerのresponse EOFでSTOP。
last AS7.999725GiB/残288KiB、RSS7.648GiB、hostavailable438.6995GiB/PSI0/OOM増分なし。full dense256x2568qUnitaryGateのgenericQSDでq1すでに約273万operation。
AS制限による割当失敗が強く支持されるが、元native error/signal/compile-vsmetrics phaseは未記録。stderrDEVNULLとformatter二次MemoryErrorによるerrorframe欠落をsource/mock2caseで確認。
SOURCE1dbdd1be2133a59ab81c7b81a6074af0880f72a3/closure62不変。6予約(2COMPLETE4RESERVED)/signal1/charge4263870068B/6identities×2ABSENT・raw18fileSHAを保存。
新science/input/compile/worker/affinity/GPU/source/caps変更・再起動0。全map成功やmemory leakは断定せず、原partialは次attemptへ混合しない。
[原因監査入口](research/track_a_h4_run07_stop_cause_audit_20261010.md)。


## 2026-10-10 H4 worker AS/RSS32GiBを明示承認・SOURCE固定

利用者の「上限を３２Gで修正して」でnewhost4workersを32GiB、driver8GiB/observer256MiB・64MiB/head16GiBを維持。両fresh/startup gateは152.25GiB。
親soft8でhardをchild準備まで保持し、workerは認可/checkout後に32、全ready後driverhard8。observerはtrusted driver PIDとnative登録workerを8/32で区別。
独立review P2 cleanup signal errorでの後続掃除skipを修正、最終53pure/mock PASS。実child/science/transpile/affinity/本体0。
SOURCE b652fff9d2907015d3b8b7b23ce8bf0fe4fce33a/closure65、旧SOURCE変更7/不変55+new3。科学/compiler/凍結入力/旧pressure/carry0/17GiB/74805/72h不変。
SOURCE/profile/input/carry/capsの準備binding、memory承認true・overalllaunchfalse/allowed[]/未seal/commandnull。fresh run/plan/auth/review/freshgatesは次回起動へ。
旧run07 one-shot/partial/cacheは保持し再利用しない。元native診断2gapsと全map完走は未検証。共有環境/venv/他job/GPU変更0。
[修正資料入口](research/track_a_h4_worker_memory32_fix_20261010.md)。


## 2026-10-10 H4 run08：worker32GiBで明示再実行

利用者が本計算再実行と開始確認後chat終了を明示指示。SOURCE8c917a5af7943969d4abede8b0bb012880efeee0/closure67、新run08/carry0。
worker4 AS/RSS32GiB、driver8GiB、observer256MiB/64MiB、admission152.25GiB、CPU2/4/5/6・driver16・observer18。科学/compiler/旧pressure/凍結6入力/17GiB/74805/72h不変。
最新run07 native6×2ABSENT/元18fileSHA/cost保持、旧partial/control/one-shotを再利用しない。新11metadata/byte tests PASS、旧53campaignと人工compile再実行0。
独立reviewとimmutable artifact・直前SOURCE/profile/input/carry/CPU/memory/pressure/OOM/容量/inode/quota/未使用lock合格後、一度mapを起動。実状態はruntime receiptへ別記録。
共有環境/既存venv/他job/GPU変更0、home内だけ。全map完走は未検証、次stage未認可。
[起動資料入口](research/track_a_h4_production_run08_20261010.md)。


## 2026-10-10 H4 run08 host1.26% STOP・限定猶予の未承認案

run08は約32分でhostPSI1.26%によりSTOP、全nonroot0/OOM増分なし/available約426GiB/role32GiB内。compile2/signal1、6 owned identities×2ABSENT、旧費用保持・次carry0。
SOURCE736bdc15ccfec6b2a715e1723850a481f0f2ff6b/closure69、旧production67不変、新inactive2files/14pure PASS。毎秒監視は維持しhost1〜5%だけ152.25GiB/nonroot0/OOM0/fresh5秒の下で最大30秒猶予、5%以上等は即STOP案。
既承認1%即STOPの変更なので利用者承認前は本番接続/再起動しない。flagsfalse/allowed[]/未seal/commandnull。科学/compiler/worker4・32GiB/共有環境/venv/GPU変更0。
[限定修正案・検査](research/track_a_h4_host_pressure_grace_fix_20261010.md)。


## 2026-10-10 H4 run09：host PSI猶予の明示承認・再実行

利用者が「猶予案を承認して再実行」と承認。host1〜5%だけnonroot0/OOM同一/available152.25GiB/fresh5秒の下で30秒猶予、5%以上・30秒継続・nonroot/OOM/欠測は即STOP、毎秒監視継続。
SOURCE251993785ef1dab2a3891bdbb5d079f5d2184f4d/closure72、新16pure回帰PASS、旧14/53/11campaign・人工compile反復0。worker4/32GiB・driver8GiB・observer256/64MiB、CPU2/4/5/6・driver16・observer18不変。
最新run08原17fileSHA/6identity×2ABSENT/cost6・complete2保持、carry0/17GiB/74805/72h/科学/compiler/凍結6入力維持。新run09、旧partial/one-shot/control/output混合なし。
限定差分review・固定artifact・直前freshgates合格後一度map、開始確認後chat終了可。共有/venv/他job/GPU変更0。
[認可・検証・起動資料](research/track_a_h4_production_run09_20261010.md)。
