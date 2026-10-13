# 研究計画資料

## 2026-10-13 mixed-access v1 source・環境準備完了／科学認可前STOP

[技術準備・数式・資源template](qc_mixed_access_source_preparation_v1_20261013.md)、[次の認可境界](../../artifacts/qc_mixed_access_source_v1/2026-10-13/launch_readiness.json)。新runner/checker、dependency wheel hash lock、22登録外synthetic PASS。full-Fock/phase/DF/adjoint/2worker保存監査を検証。
`src/trottertracks/representation_exploration/qc_mixed_access_{math,resource,execution,pipeline,checker}_v1.py`、`scripts/run_qc_mixed_access_feasibility_v1.py`、`scripts/verify_qc_mixed_access_feasibility_v1.py`、`scripts/check_qc_mixed_access_{synthetic,preparation}_v1.py`、`tests/test_qc_mixed_access_feasibility_v1.py`、artifacts `artifacts/qc_mixed_access_source_v1/2026-10-13/`。
固定H4/7法/T1,4/r1…1024の契約は不変。今回分子生成・SCF・登録科学PF・compile・GPTレビュー0、function/full energy費用UNKNOWN。次はユーザーの固定一回実行承認。技術準備だけでGPT再レビューを要求しない。旧122 entries/科学source/認可/STOPとglobal description欠落FAIL、N1STOP/N2HOLD/N3CLOSEDを保持。以下は過去milestone。

## 2026-10-13 N3成果確定・現検証系列終了

[公開・remote確認完了](n3_saved_attribution_v1_claim_and_observability_20261013.md)：変更16blob/証拠14blob/追加リンク38件確認。現行N3の文書作業完了、科学STOP。

[確定claim・成立条件・再開条件](n3_saved_attribution_v1_claim_and_observability_20261013.md)、[理論claim map](n3_theory_claim_map_v1_20261012.md)、[完了GPTレビュー原本](n3_saved_attribution_v1_scientific_review_2026-10-13.md)。
証拠は `artifacts/n3_achievements_closure/2026-10-13/existing_evidence_index.json` の固定commit参照。研究判断は同directoryの `research_decision.json`。
現N3の追加benchmark・保存帰属診断系列を終了し、主研究としての採択を見送る。理論と厳密分離・有限prototype観測を条件付きで保持。具体的な有用taskまたは理論差分でのみ再検討する。
文書整理のみ、新science/数値的事後評価/再認証/benchmark/テスト/再レビュー0。元source/result/primary/gate/STOP・消費済み認可を保持。N1停止/N2保留、中心テーマnull、新テーマ未採択、Mandatory STOP。
同じ内容の科学レビューは完了済みで、文書整理だけで再要求しない。全体manifestの既存description欠落FAILを個別保存照合と区別し、無断修正しない。以下の旧stage記述は当時の履歴。


## 2026-10-13 N3保存帰属v1：完了レビューを採用、数値事後評価の認可前

[準備資料](n3_saved_attribution_v1_preparation_20261013.md)、[限定契約](n3_saved_attribution_v1_posthoc_contract_20261013.md)。
元6H（非負density、full5/6・ν2/3、一外部mode、候補2mode/H、元ε=.01、geometry/basis/DF/L_D/PF窓なし）の216 path/432event/696候補をliteral/pointer索引化した。
18 inaccurateは全てSOS/V2_banded_6/external5。修復前proposal/r・専用refutation timerは未保存。既存3証拠を丸めず接続し、原監査を参照。
新source・gate・posthoc runner、synthetic23 PASS、2spawnとserialの合成一致。登録データの新数値事後評価・PSD再認証0。
レビュー第12.6節に従い、次は固定source/registry/planの保存事後評価一回の明示認可。通常技術作業ごとのGPT詳細レビューは不要。
新batch/新入力/新ε/solver変更/再試行0、旧source/results/STOP、N1停止/N2保留、中心テーマnullを保持。以下は過去milestone。

## 2026-10-13 N3幅比較v2：固定6H pilot完了・GPT科学判断待ち

[一次結果](n3_width_comparison_v2_results_20261013.md)、[GPTへの引継ぎ](n3_width_comparison_v2_results_review_handoff_20261013.md)。
3stress family×full5/6、ν2/3、非負density、一外部mode、候補2mode/H、ε_proj=.01（化学geometry/basis/DF/L_D/PF窓なし）。
source3eba0814で一回/retry0、504 mode-method records完了。独立saved auditは1224 certificate・36反証record PASS。
NORM/OCC/同値SDP/SOSは6/6H・12/12modeで目標受理、追加H縮約0。OCCの有限幅はcheap ORよりH5/6で狭く、同値SDPより全6Hで狭い幅・短いcold medianを取得した。
新しい科学結果の研究価値は未採択。認可消費済み、Mandatory STOP、次はGPT科学レビューの開始承認待ち。N1停止/N2保留、旧primary/gate/source/resultsを保持。以下は過去milestone。

## 2026-10-12 N3 OLD-family v1 one-shot results

[固定S1二例の有理dual分離・独立監査・STOP](n3_old_family_v1_results_20261012.md)。同じ保存U/c/OCC有理δでNORM＋全旧κ/μ族（FIVE包含）を分離。
source/module/runner/checker/testsは凍結済み準備版のまま。artifacts `artifacts/n3_old_family_v1/2026-10-12/diagnostic1/` と `execution_audit/`。
一回/retry0、posthoc development、元primary/gate/STOP不変、GPTの新結果判断へ。以下は過去milestone。

## 2026-10-12 N3 OLD-family v1 preparation
+
+[数学契約・実装・実行前STOP](n3_old_family_v1_preparation_20261012.md)。
+modules `n3_old_family_contract_v1` / `n3_old_family_v1` / `n3_old_family_check_v1` / `n3_old_family_inputs_v1`、scripts prepare/run/verify_n3_old_family_v1、tests test_n3_old_family_v1。
+artifacts `artifacts/n3_old_family_v1/2026-10-12/`。23 synthetic PASS、新S1診断0、別実行指示待ち。元結果/STOP不変。以下は過去milestone。

## 2026-10-12 N3 saved-width v1 evaluation complete

[一回結果・scope・費用](n3_saved_width_v1_results_20261012.md)：有限保存集合のOCC strict2/8（S1の1base）、BASE strict6/8、有理独立saved audit全8PASS。
module/runner/checker/testsは準備sourceのまま、artifacts `artifacts/n3_saved_width_v1/2026-10-12/evaluation1/` と `evaluation_audit/`。
posthoc development、元primary/2-base未達不変、Mandatory STOP、次段未認可、中心テーマnull。以下は過去milestone。

## 2026-10-12 N3 saved-width v1 preparation

[証拠整理・式・synthetic・数値適用前STOP](n3_saved_width_v1_preparation_20261012.md)。
module `n3_saved_evidence_v1.py`/`n3_saved_width_v1.py`/`n3_saved_width_analysis_v1.py`; scripts `prepare_n3_saved_width_v1.py`/`run_n3_saved_width_v1.py`/`verify_n3_saved_width_v1.py`;
tests `test_n3_saved_width_v1.py`; artifacts `artifacts/n3_saved_width_v1/2026-10-12/`. 実8入力の新幅は未評価。


## 2026-10-12 N3 OCC v1 fixed eight-input result

[一回結果・監査・STOP](n3_occupation_transfer_v1_results_20261012.md)。強対照/OCC各8/8、固有gain0、保存check5,008,702 PASS。
Frozen generator/runner/checker/tests are unchanged; artifacts `artifacts/n3_occupation_transfer_v1/2026-10-12/run1/` and `execution_audit/`.


## 2026-10-12 N3 OCC v1 implementation preparation

[実装・検証・実行前STOP](n3_occupation_transfer_v1_source_preparation_20261012.md)。新8入力batch未認可。
module `src/trottertracks/representation_exploration/n3_occupation_transfer_v1.py`;
independent `n3_occupation_transfer_check_v1.py`; runner `scripts/run_n3_occupation_transfer_v1.py`;
saved verifier `scripts/verify_n3_occupation_transfer_v1.py`; tests `tests/test_n3_occupation_transfer_v1.py`;
artifacts `artifacts/n3_occupation_transfer_v1/2026-10-12/preparation/`.


## 2026-10-12 N3 occupation certificate preparation

[数学検証・比較設計](n3_occupation_redesign_mathematical_audit_20261012.md)。新科学batch未認可。
module `src/trottertracks/representation_exploration/n3_occupation_certificate_audit.py`;
runner `scripts/verify_n3_occupation_redesign_preparation.py`;
coefficient-only preparation `scripts/prepare_n3_occupation_transfer_inputs.py`;
test `tests/test_n3_occupation_certificate_audit.py`;
artifacts `artifacts/n3_occupation_redesign_preparation/2026-10-12/`.


2026-10-11 完了済みN13科学レビューを受領し、N3保証生成とN1decode復旧を別契約へ具体化した。
[次作業の入口](../../docs/research/n3_generator_n1_recovery_preparation_20261011.md)。
N3は一外部・実one-body・非負densityの一変数LMI案と未使用12有理係数入力を結果前固定。
N1は元44 factor/178 pair・q/精度/対照を保つdecode修正・N1のみ一回実行のscope。N2は完了扱い。
今回は文書・係数literal・metadata/hash検査のみ、実装・新科学tests・solver/対角化/compile0。
準備監査 `scripts/verify_n3_generator_n1_recovery_preparation.py`、artifacts `artifacts/n3_generator_n1_recovery/2026-10-11/preparation/`。
future module/runner/tests/checkerは契約JSONに記した未実装path。旧source/failure/結果は不変。
`PREPARATION_ONLY_MANDATORY_STOP`、新batch未認可、中心テーマnull。以下は過去milestoneの履歴として保持する。


2026-10-11 現在の次段準備は[N1/N3再設計の限定scope v1](n13_redesign_limited_scope_v1_20261011.md)と[machine contract](n13_redesign_limited_scope_v1_20261011.json)。
[提供科学レビュー](n13_redesign_inputs/n123_feasibility_scientific_review_2026-10-11.md)は完了済み。N1/N3を優先しN2を固定S修正に限定するが、中心テーマは採択していない。
設計だけを具体化し、新science/compile/LP/対角化/tests0。`DESIGN_PREPARED_SCIENCE_NOT_AUTHORIZED`でSTOP。
[判断記録](研究ノート/2026-10-11_n13_redesign_preparation.md)。以下は旧stageの当時の履歴。

2026-10-11 N1/N2構成・N3係数下界の限定feasibilityを完了。
[結果・GPT判断事項](hamiltonian_algorithm_design_results_20261011.md)、[固定scope](hamiltonian_algorithm_design_scope.md)を入口とする。
N1：JW4・ν2・signed DF rank2・決定論S2、T=.6、delta=.6/.3/.15/.075、εH<=.02。
N2：occupation4・full-ij J・k<=2 signed charge、T=.7、係数予算.0021。共通εcomplex=.04。
N1はframe費用を減らせたが元H精度のshot負担でexactより高く、N2は加算込みでdirectより高い。
N3：6 modes・ν2・初期active4、gapped例で係数δ=.001549636、小gapではfull spaceへ戻す。
science source ed335a0、85 local tests、280 native IR/120 rows/177 source blobsの保存照合PASS。immutable CIではない。
module `src/trottertracks/representation_exploration/algorithm_design.py`、runner `scripts/run_hamiltonian_algorithm_design.py`、
保存集約器 `scripts/finalize_hamiltonian_algorithm_design.py`、verifier `scripts/verify_hamiltonian_algorithm_design.py`、
構成・選択・保存監査 `scripts/audit_hamiltonian_algorithm_design_construction.py`、
tests `tests/test_hamiltonian_algorithm_design.py`、artifacts `artifacts/hamiltonian_algorithm_design/2026-10-11/`。
中心テーマnull・次段false・mandatory STOP、GPT判断待ち。以下の旧stage/他Trackの履歴・契約・STOPは保持する。

2026-10-11整理：A寄与分解・B′物理pairの限定batch（計算系列日付2026-10-10）を完了。
[結果とGPT判断事項](representation_attribution_pair_results_20261010.md)、[scope](representation_attribution_pair_scope.md)を入口とする。
Aは3-mode JW全Fock8、synthetic exact DF rank2、native L_D0/1/2＋whole-Pauli＋旧/完全占有core、T=.4、delta=.4/.2/.1。
Bはphysical3+aux1 isometry fixture、JW、L_D0/r1/K2、T=.2、delta=.2/.1/.05。geometry/化学basis/分子fittingなし。
費用は固定q1、meanはq1/2/4、epsilon=.05/.02、full-physical-Fock bias診断で揃える。
Aは安いwhole-Pauliを加えるとcore改善を支持せず、完全占有coreの追加効果はRZ約.452%減・CX約.202%増。
B′はlambda/shot数を下げてもbasis/native費用でdirect-Pauliより高い。C追加scan0。
source3975650、85 focused local tests、1558 compiles、177 source/input blobsと全IRの別保存監査PASS。
local開発証拠でimmutable CI/外部科学再現/新規性認定ではない。
`ATTRIBUTION_PAIR_COMPLETE_AWAITING_GPT_REVIEW`、中心仮説null、次段false、mandatory STOP。
以下の旧stage/Track A/Bの履歴・science/契約/STOPを変更しない。


2026-10-10 A寄与分解・B′物理pair限定batchの[固定scope](../../docs/research/representation_attribution_pair_scope.md)。
module `src/trottertracks/representation_exploration/attribution_pair.py`、runner `scripts/run_representation_attribution_pair.py`、
tests `tests/test_representation_attribution_pair.py`、保存監査 `scripts/verify_representation_attribution_pair.py`、
artifacts `artifacts/representation_attribution_pair/2026-10-10/`を一組に読む。旧系列は当時の履歴として保持する。


2026-10-10の独立系列最新は
[限定構成・同一信号精度比較結果とGPTへの判断事項](representation_construction_comparison_results_20261010.md)。
[固定scope](representation_construction_comparison_scope.md)と
[提供独立review/ZIP inputs](representation_construction_inputs/)を区別する。
3-mode rank2 core生成とnative全endpoint比較、2/3qubit Isingの対称S2/THRIFT比較、isometry構造接続を完了。
source ce99b57、374 compiles、70 tests、保存監査PASS。local開発証拠、中心仮説null、次段false、mandatory STOP。
旧初期結果・Track A/Bは変更しない。以下は各milestone当時の履歴。

2026-10-10、別branchの[Hamiltonian表現探索初期結果](representation_exploration_initial_validation_20261010.md)を追加。
[scope](representation_exploration_scope.md)と
[利用者入力/GPT初期レビュー](representation_exploration_inputs/)を区別する。
正のfactor混合、生成子反射、THRIFT mixed oracleをtoyで検証。中心仮説未採択、GPTレビュー待ちSTOP。
既存Track A/Bの契約・科学source・STOPは変更しない。以下の状態記述は基点までの履歴。

2026-10-06のTrack A入口は[サーバー側への正式契約準備指示](gpu_server_track_a_h4_geometry_contract_preparation_prompt.md)と
[確認済みscope JSON](track_a_h4_geometry_contract_preparation_scope_v1.json)。
server preparation `c2ab34f`を基に追加6距離・最大12 workersを契約条件とし、218 template/点・74,784 wrapper上限を記録する。
checkpoint数値回路identityと合成record検査を補強する。本計算・science source port・追加benchmarkは未認可。
旧証拠・原稿・Track B不変。利用者指示で今回の契約準備資料だけをcommit/pushする。以下は各milestone当時の履歴。

現在は[Track A H4 geometryと要求精度の拡張設計案](track_a_geometry_precision_extension_proposal_v0.md)と
[別JSON](track_a_geometry_precision_extension_proposal_v0.json)を優先する。
[GPUサーバー側準備指示](gpu_server_track_a_h4_geometry_resource_preparation_prompt.md)はserver既存環境を優先し、
科学計算を開始せずinventory・synthetic CPU benchmark・契約草案までを依頼する。
利用者の新しい指示により原稿作成を一旦保留。追加6点候補・218 templateの資源mapを提案したが、
距離・host・worker・新実行source/authorizationは未固定。科学STOP、本計算・性能benchmark未実行。
既存原稿と結果を保持し、H6/H8や追加96は今回に含めない。以下の収束判断は当時の履歴。

2026-10-05の最新状態は[Track A通し原稿v0.1](../manuscripts/track_a_resource_study_v0_1.md)。
[補足](../manuscripts/track_a_resource_study_supplement_v0_1.md)、
[claim audit](../manuscripts/track_a_resource_study_claim_audit_v0_1.md)、
[独立review依頼](../manuscripts/track_a_publication_readiness_review_request_v0_1.md)を分離した。
4主図・補足図1は保存値からの再表示で、新科学計算0。研究方針・旧result/manifestは変えず科学STOP。
以下の設計のみ・未実施記述は各milestone当時の履歴である。

PM-2後の方針reviewを採用し、Track Aを現在の証拠で閉じる限定資源比較研究として原稿化する。
[主張・証拠対応表](track_a_post_pm2_claim_evidence_map.md)に固定commit・証拠階層・先行研究比較、
[原稿設計](track_a_post_pm2_manuscript_design.md)に章構成・主要4図・補足・完成確認をまとめた。
今回は文書設計まで。図生成・通し原稿は未実施、新科学計算0、旧status/manifest不変。
PM-3やTrack B統合は認可せず、以下の準備・review待ち記述は各stage当時の履歴として保持する。

最新は[PM-2保存値解析結果](../pr2_pm2_precision_resource_result_validation.md)。
固定302点・67,346行とε=0.05の元結果再現を照合し、mandatory STOP・研究方針review待ち。
POSTHOC local evidence、新しい科学計算0。利用者指示でresult commitへ収録する。以下の準備記述は当時の履歴である。

PM-2の[保存値解析実装](pr2_pm2_precision_analysis_implementation.md)をsource `324435d77b6642dbd44e8d1f178420daf62e77ed`で固定した。
62 synthetic-only tests合格、real numerical gate/precision map未実行、利用者の明示解析指示待ち。
[固定契約](pr2_pm2_precision_resource_contract_v1.md)は変更せず、準備stageの記述は当時の履歴として読む。

2026-10-05のPM-1後reviewを採用し、[PM-2精度と測定込み資源境界契約](pr2_pm2_precision_resource_contract_v1.md)を固定した。
全218候補とM2元5構成、POSTHOC、軸別費用、ε範囲、共通P、schema/input identity、全status後STOPが対象。
解析は未認可で、保存JSON/input coverageだけを確認した。現行RQの限定と完成条件はこの契約を優先する。

PM-1は最終review承認・明示launch後、一回だけ実行し[結果を照合](../pr2_pm1_discard_result_validation.md)した。
H4 linear 1.00 Å、STO-3G、DF rank12、T=0.8、B0 rank4/5 × q=1/2/4/8が全件accuracy適格。
8 signals/16 wrappers、source134不変、pre/post201 local tests passed、fail/skip0。
新B0最小rank5・q1は旧rank6よりprimary RZ9.34%減だが、保存B2 r4点推定の1.75659倍。
`PM1_DISCARD_MAP_COMPLETE_AWAITING_REVIEW`でmandatory STOP、次段未認可・研究判断null。
利用者指示でlocal executionの結果と監査をresult commitへ収録し、次は研究方針reviewへ戻る。
以下の準備・未認可記述は各milestone当時の履歴であり、この最新節と区別する。

post-M2方針reviewに基づく[PM-0証拠帰属解析](pr2_post_m2_evidence_attribution.md)を完了した。
現在のA claim/RQの限定は[PM-0報告](pr2_post_m2_evidence_attribution.md)と[研究概要](研究概要・現状.md)を参照する。
旧S2からのq-only attribution、selector primary損失、異なる候補集合のP感度を限定し、PM-1以降は未認可。
続く[PM-1近接discard契約](pr2_pm1_nearby_discard_contract_v1.md)は契約・science source・synthetic testsまで固定した。
local source commit `fd7552e`とsealed preparation planに結合するが、実行認可ではない。
development B0 rank4/5の8候補、16 wrapperを超えず、次は別authorization/reviewでSTOPする。
[GPTへの準備bundleレビュー依頼](pr2_pm1_preparation_external_review_request_fd7552e.md)は、
authorizationを作る段階へ進めるかだけを確認し、科学実行を認可しない。

このディレクトリには、DF部分ランダム化・有限RTE・RPE compiled cost研究の目的、背景、未解決点、解析手順および数値評価計画をまとめる。

実装、runner、テスト、成果物を含むリポジトリ全体の区分は
[`../../PROJECT_MAP.md`](../../PROJECT_MAP.md)を参照する。

PR-2の最新系列は、旧STOPを維持した別snapshotでV1–V3、V4 correctness、development-only S2を完了した。
S2後reviewでは新手法claimを外し、matched-accuracy resource-map研究へ狭めた。
[M1前先行研究gate](pr2_matched_accuracy_prior_art_gate_v1.md)は`PROCEED_RESOURCE_STUDY`、
[M1実装契約](pr2_matched_accuracy_m1_implementation_contract_v1.md)は208候補台帳、最大4境界候補、
16-cell selector、schema、zero-compute guardを固定した。続く
[M1前最終amendment](pr2_matched_accuracy_m1_preexecution_amendment_v2.md)は追加prior-art二件を照合し、
`SELECTION_LIMITED`ならM1-B compile前に停止するhard barrierを固定した。M1-A v1実行は固定Kのconfig
整合性でresult作成前に停止し、[v1.1再認可](pr2_matched_accuracy_m1_execution_authorization_v1_1.md)が
同じ候補のcompile-free再実行を許可した。再実行は210候補を評価したが、64 proxy-frontier候補中
52件が16-cell cap外に残り、`SELECTION_LIMITED`で停止した。
外部reviewは`PROCEED_BOUNDED_COMPILE_EXPANSION`を選び、
[M1-B1 bounded compile契約](pr2_matched_accuracy_m1_b1_bounded_compile_contract_v1.md)とzero-compute planが
random 194 cell、baseline 16 cell、計12,448 wrapperのidentityを固定した。
[execution contract amendment v2](pr2_matched_accuracy_m1_b1_execution_contract_amendment_v2.md)は、外部reviewの
修正要求に従いactual execution sourceをauthorizationより先に固定し、science runnerをcompile map完成・
review待ちで停止させる。研究四分岐はrunnerが自動選択しない。
[execution authorization v1](pr2_matched_accuracy_m1_b1_execution_authorization_v1.md)はsource commit `33f436b`と
planを固定して一回のM1-B1を認可した。12,448 wrapper mapは完了し、
[M1-B1結果検証](../pr2_matched_accuracy_m1_b1_result_validation.md)はactual six-metric ParetoをB2 rank 3、q=1の
2件と確認した。判断は`CONTINUE_RESOURCE_STUDY`で、その時点ではheld-outと追加96は未承認だった。
[M2 held-out transfer契約](pr2_matched_accuracy_m2_held_out_transfer_contract_v1.md)は、その2件とB0/B1/B3代表の
計5構成、primary RZ、6指標Pareto、10% materiality、重大underestimate、4 terminal status、196-wrapper上限を
zero-compute固定した。外部reviewの修正要求は
[usable B2 amendment v2](pr2_matched_accuracy_m2_transfer_contract_amendment_v2.md)へ反映した。
修正版契約と正式planをcommit固定し、[science実装](pr2_matched_accuracy_m2_transfer_execution_implementation.md)を
追加し、source/planを固定した後の別authorization・最終review承認・利用者指示を経てM2を一度実行した。
最新の[M2結果照合](../pr2_matched_accuracy_m2_transfer_result_validation.md)は`TRANSFER_SUPPORTED`。
固定5構成、196 wrapper、B2二件のParetoとratioを確認したlocal execution evidenceをcommit保存し、一般的method最適性とはしない。
現在はmandatory STOP、研究方針全面review待ち。追加96、retuning、S3等を認可しない。

初めてこの研究を確認する場合や、Codexを使って発表・共有資料を作る場合は、まず
[研究概要・現状](研究概要・現状.md)を読む。この一冊で現在の研究段階、採用済みの
方針、主要な検証結果、未決定事項および証拠の読み順を確認できる。

## 文書の役割

| 種類 | 正本とする内容 |
|---|---|
| [研究概要・現状](研究概要・現状.md) | 現在地と資料作成用の統合要約 |
| [事前検証カタログ実施証拠索引](prevalidation_catalog_evidence_map.md) | 実施したカタログID・work packageとGitHub上の証拠の対応 |
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
| 4 | [P-A joint synthesis先行研究監査](pa_joint_synthesis_prior_art_audit.md) | P-A v1の検索範囲、closest prior art、限定novelty statement、v2との境界を記録する。 |
| 5 | [P-A blind transfer事前登録](pa_joint_synthesis_blind_validation_preregistration.md) | 実行前にH5 physical transferとH4 compiler transferの条件・gateを固定した記録。 |
| 6 | [P-A blind transfer検証結果](../research_direction_joint_synthesis_blind_validation.md) | 両stratumの完了、固定gate、結果、判断、scopeを記録する。 |
| 7 | [P-A v1形式化・機構監査](pa_joint_synthesis_v1_formalization.md) | 目的関数、DP、計算量、同値性条件と、完成済み証拠が一run一segment・order 0へ退化している制約を記録する。 |
| 8 | [P-A非退化mechanism validation事前登録](pa_joint_synthesis_mechanism_validation_preregistration.md) | 明示的一区間baseline、forced support変化、order 2 stream、固定gateと停止規則を計算前に記録する。 |
| 9 | [P-A非退化mechanism validation結果](../research_direction_joint_synthesis_mechanism_validation.md) | 30/30 taskで分割・plan差・追加RZ改善が0だった結果と、P-A停止・P-C復帰判断を記録する。 |
| 10 | [P-C geometry tracking・breakdown事前登録](pc_geometry_tracking_breakdown_preregistration.md) | 8 geometry、2 policy、blind region、4 pair、7 gate、固定停止規則を計算前に記録する。 |
| 11 | [P-C geometry tracking・breakdown結果](../research_direction_geometry_tracking_breakdown.md) | 16/16 task、prefix変更0、stretch予測破れ、current H4 family停止を記録する。 |
| 12 | [P-D energy・tail Pareto事前登録](pd_energy_tail_pareto_preregistration.md) | 固定5公式、`L_D=3` development、`L_D=4` blind、7 gateと停止規則を記録する。 |
| 13 | [P-D energy・tail Pareto結果](../research_direction_energy_tail_pareto.md) | energy-onlyとtail-awareのblind逆転、条件付き候補判断、後続の現実化gateを記録する。 |
| 14 | [P-D現実化Go/No-Go事前登録](pd_realization_go_no_go_preregistration.md) | 負時間D1、内部`H_D` D2、fresh `L_D=5` D3、固定閾値、終了後の停止規則を記録する。 |
| 15 | [P-D現実化Go/No-Go結果](../research_direction_pd_realization.md) | D1--D3全通過、正式主研究候補化、未評価範囲、研究再設計の停止点を記録する。 |
| 16 | [P-D主研究契約](pd_primary_research_contract.md) | S0の主RQ、固定H4 scope、B0/B1a/B1b/B2/B4、Case A--D、S1後の強制停止を固定する。 |
| 17 | [P-D既知baseline](pd_prior_art_and_baselines.md) | 既知のabsolute-tail-time modelとS1で新たに問うfinite/internal/construction差を分離する。 |
| 18 | [P-D S1事前登録](pd_s1_fair_comparison_preregistration.md) | 共通時間・位相予算、nested/native、regret、K4 trigger、境界規則を結果前に固定する。 |
| 19 | [P-D S1結果](../research_direction_pd_fair_comparison.md) | B1b/B2/B4一致、Case C/D不成立、B1a上限依存によるCase B＋undetermined停止を記録する。 |
| 20 | [P-D S1事後再解析計画](pd_s1_posthoc_reanalysis_plan.md) | 固定artifact、主baseline、5%近傍、B1a・構成差の再集計規則を事後計画として固定する。 |
| 21 | [P-D S1事後再解析結果](../research_direction_pd_s1_posthoc.md) | 一次Case Bを保存し、主baselineのCase A相当解釈、B1a診断、P-D停止を記録する。 |
| 22 | [R3先行研究監査・条件付き最小契約](r3_prior_art_and_minimal_contract.md) | 広いR3のNo-Go、R3-S0の方法論監査不通過、実行しない条件付き最小契約を記録する。 |
| 23 | [finite-RTE位相・信号半径分離契約](finite_rte_phase_amplitude_contract.md) | FR-0の演算子定義、非可換積の位相・半径境界、norm baseline、主張範囲を固定する。 |
| 24 | [finite-RTE位相・信号半径分離の先行研究監査](finite_rte_phase_amplitude_prior_art.md) | finite LCU/RTE、PF spectral解析、randomized平均信号、phase-lag、nonunitary分解との限定差分を記録する。 |
| 25 | [FR-1非可換toy機構試験 事前登録](finite_rte_phase_amplitude_fr1_preregistration.md) | 2×2 toyの固定grid、available-information baseline、G0--G4、FR-1後の強制停止を定める。 |
| 26 | [FR-1 finite-RTE位相・信号半径分離結果](../finite_rte_phase_amplitude_validation.md) | 33条件・99状態、G0/G1/G3/G4通過、G2不通過、`GO_FR2_MECHANISM_ONLY`とFR-2停止を記録する。 |
| 27 | [FR-1後の研究方針改訂案](../../fr1_revised_research_plan_20260926.md) | involution toyの特殊性、正scalar分離、非一様4×4案をまとめた再設計入力。正式契約と衝突する場合は次項を優先する。 |
| 28 | [FR-R0正scalar分離・構造比較契約](fr_revision_scalar_structure_contract.md) | 正scalar、I0/I1/I2、共通$\gamma$/最適化baseline、FR-R1事前登録要件、定量的GO/STOPを固定する。 |
| 29 | [FR-R1a正scalar事後再解析計画](fr_revision_fr1a_posthoc_plan.md) | 既存33条件・99状態の固定artifactだけを説明監査し、旧G2とdecisionを変更しないposthoc規則を定める。 |
| 30 | [FR-R1b非一様4×4事前登録](fr_revision_nonuniform_preregistration.md) | 20条件・61状態行・2 semantic control、固定状態、強いbaseline、R0--R7と強制停止を結果前に固定する。 |
| 31 | [FR-R1a正scalar事後再解析結果](../fr_revision_fr1a_posthoc.md) | 33条件・99状態を再構成し、891 method recordのsoundness違反0、片側FR認証差0、`POSTHOC_SCALAR_EXPLAINS_OLD_GAIN`を記録する。 |
| 32 | [FR-R1b非一様4×4結果](../fr_revision_nonuniform.md) | 20条件・61状態で非一様FR固有のstrict gain 8件を確認したが、固定予算の片側認証差0により`MECHANISM_ONLY_NO_PRACTICAL_GO`で停止した結果を記録する。 |
| 33 | [FR研究主張・証明義務・完成原稿契約](fr_research_claim_and_manuscript.md) | C1/C2を中核、C3を条件付き応用に置き、既知事項、新規性監査、T0--T6、追加計算を始めない完成判定を固定する。 |
| 34 | [PR-2／PR-3最小pilot事前登録](pr2_pr3_minimal_pilot_preregistration.md) | PR-2/PR-3の各一条件、correctness、GO/STOP、pilot後の強制停止を結果前に固定する。 |
| 35 | [PR-2／PR-3最小pilot結果](../pr2_pr3_minimal_pilot_validation.md) | PR-2 `GO_PR2`、PR-3停止、PR-2主題候補選択、one-step/toy scopeを記録する。 |
| 36 | [PR-2主研究契約](pr2_primary_research_contract.md) | 限定的な資源研究claim、prefix identity gate、rank-6 anchor、独立条件、完成・停止規則を固定する。 |
| 37 | [PR-2 S1--S3結果前事前登録](pr2_s1_s3_preregistration.md) | matched baseline、finite-RTE、signal、shot、compiled cost、stage stopと実装gapを固定する。 |
| 38 | [PR-2 S0/S1前 独立批判レビュー](pr2_s0_s1_external_review_d3e1723.md) | fixed commit `d3e1723`を監査し、`AMEND_BEFORE_S0`と判定した外部レビュー。 |
| 39 | [PR-2 S1--S3事前登録 amendment v2](pr2_s1_s3_preregistration_amendment_v2.md) | normalization補正、shot overhead、baseline family、S1 correctness-onlyとレビュー停止を結果前に固定する。 |
| 40 | [PR-2 S0/S1実行許可 amendment v3](pr2_s0_s1_execution_amendment_v3.md) | S0と、S0通過時だけのS1 correctnessを許可し、S1後mandatory STOPを固定する。 |
| 41 | [PR-2 S0 input reproduction停止報告](pr2_s0_reproduction_stop_c644925.md) | pilot development hash不一致で`STOP_INPUT_REPRODUCTION_MISMATCH`。S1を実行せず、近似一致でgateを緩和しない。 |
| 42 | [PR-2 S0 STOP後 外部レビュー依頼](pr2_s0_external_review_request_c644925.md) | terminal STOP確認、canonicalizationの最低要件、終了または新pilot化をGPTへ批判的レビューさせる指示。 |
| 43 | [PR-2 V4/S2 development実行許可](pr2_v4_s2_development_authorization_v5.md) | V4 correctness、rank 6 primary比較、rank 3/9 control、held-out未開封、S2後mandatory STOPを結果前固定する。 |
| 44 | [PR-2 V4/S2 development結果](../pr2_v4_s2_development_validation.md) | B2/B3 frontier、rank control、実行範囲、held-out前の研究方針review判断を記録する。 |
| 45 | [PR-2 matched-accuracy M1前先行研究gate](pr2_matched_accuracy_prior_art_gate_v1.md) | 最接近研究とのclaim overlapを固定し、新手法ではなく限定resource studyとして`PROCEED_RESOURCE_STUDY`と判定する。 |
| 46 | [PR-2 matched-accuracy resource-map契約](pr2_matched_accuracy_resource_contract_v1.md) | baseline、可変q correctness、random direct-compile 16-cell選抜、`SELECTION_LIMITED`、held-out前の別freezeを固定し、M1計算を未承認のまま保つ。 |
| 47 | [PR-2 matched-accuracy M1実装契約](pr2_matched_accuracy_m1_implementation_contract_v1.md) | 208候補identity、最大4境界候補、occurrence seed、16-cell selector、result schema、zero-compute dry-runを固定し、M1科学計算を未承認のまま独立reviewへ渡す。 |
| 48 | [PR-2 matched-accuracy M1前最終amendment](pr2_matched_accuracy_m1_preexecution_amendment_v2.md) | 2026年の近接研究二件をclaim単位で追加照合し、M1-Aで`SELECTION_LIMITED`ならcompile job 0のままmandatory STOPするM1-B前hard barrierを固定する。 |
| 49 | [PR-2 matched-accuracy M1-A実行認可 v1](pr2_matched_accuracy_m1_execution_authorization_v1.md) | development-only、最大212 signal、compile 0、held-out access 0を固定した初回認可。実行はconfig整合性でresult作成前に停止した。 |
| 50 | [PR-2 matched-accuracy M1-A再認可 v1.1](pr2_matched_accuracy_m1_execution_authorization_v1_1.md) | 固定KのRTEConfig self-consistencyだけを修正し、候補・threshold・selectorを変えず同じM1-Aを再認可する。 |
| 51 | [PR-2 matched-accuracy M1-A結果](../pr2_matched_accuracy_m1_a_validation.md) | 210候補、64 proxy frontier、52未選択frontierにより`SELECTION_LIMITED`でcompile 0停止した結果を記録する。 |
| 52 | [PR-2 M1-A limited後 GPTレビュー依頼](pr2_m1_a_selection_limited_external_review_request_3c1831e.md) | commit `3c1831e`を固定し、bounded compile拡張・technical note縮小・研究停止の三択と回答形式を指定する。 |
| 53 | [PR-2 matched-accuracy M1-B1 bounded compile契約](pr2_matched_accuracy_m1_b1_bounded_compile_contract_v1.md) | 旧selector監査を保存し、M1-A適格random 194 cell×32 trajectory×2軸とB0/B1 16 cell×2軸の12,448-wrapper有限grid、cache identity、B1後STOPをzero-compute固定する。 |
| 54 | [PR-2 M1-B1実行前GPTレビュー依頼](pr2_m1_b1_preexecution_external_review_request_1228168.md) | source commit `1228168`とplan fingerprintを固定し、別result-prior execution authorizationへ進む前の候補・seed・cache・resource・停止規則レビューを依頼する。 |
| 55 | [PR-2 M1-B1 execution contract amendment v2](pr2_matched_accuracy_m1_b1_execution_contract_amendment_v2.md) | actual science module/runner/testを先にsource commit化し、result statusをcompile map完成review待ちまたはimplementation failureだけへ限定する。 |
| 56 | [PR-2 M1-B1 execution authorization v1](pr2_matched_accuracy_m1_b1_execution_authorization_v1.md) | source commit `33f436b`とplan v2を結び、12,448 wrapper、最大6 workers、一回限り、held-out/追加96/研究自動判定禁止を結果前固定する。 |
| 57 | [PR-2 M1-B1 authorization最終GPTレビュー依頼](pr2_m1_b1_execution_authorization_external_review_request_8fc2400.md) | bundle commit `8fc2400`を固定し、source-first順序、seed/cache/checkpoint、resource cap、2 terminal statusを本計算前に最終確認する。 |
| 58 | [PR-2 M1-B1 GPUサーバーCPU高速化確認依頼](gpu_server_pr2_m1_b1_cpu_acceleration_feasibility_prompt.md) | 実runを変更せず、科学データ0・GPU 0のsynthetic CPU benchmarkだけで移行価値を判定する運用依頼。 |
| 59 | [PR-2 M1-B1 actual compile map結果](../pr2_matched_accuracy_m1_b1_result_validation.md) | 210 cell、12,448 wrapper、全checkpoint/cacheを再検査し、B2 rank 3、q=1のactual frontierと`CONTINUE_RESOURCE_STUDY`、held-out前STOPを記録する。 |
| 60 | [PR-2 M2 held-out transfer契約 v1](pr2_matched_accuracy_m2_held_out_transfer_contract_v1.md) | developmentで固定した5構成だけをH4 1.30 Åへ移すため、primary/secondary判定、10% materiality、重大underestimate、4 terminal status、196-wrapper上限、全status後STOPをzero-compute固定する。 |
| 61 | [PR-2 M2 transfer契約GPTレビュー依頼](pr2_m2_transfer_contract_external_review_request_06b2c32.md) | source commit `06b2c32`、plan hash/fingerprint、5構成、判定、resource capを固定し、science source実装前の独立reviewを依頼する。 |
| 62 | [PR-2 M2 usable B2 amendment v2](pr2_matched_accuracy_m2_transfer_contract_amendment_v2.md) | v1/draftを保存し、Pareto support・ratio・NOT_SUPPORTED/INCONCLUSIVEを同じusable B2集合へ統一したcommit固定済み結果前修正。契約自身は科学実行を認可しない。 |
| 63 | [PR-2 M2結果照合](../pr2_matched_accuracy_m2_transfer_result_validation.md) | 固定5構成のH4 1.30 Å transfer、196 wrappers、source/manifest/checkpoint・pre/post tests照合、`TRANSFER_SUPPORTED`後のmandatory STOP。 |

## 研究ノート

日付ごとの実装方針、判断理由、検証結果および次の課題は、
[研究ノート](研究ノート/README.md)に時系列で記録する。研究ノートは変更の
経緯を残すための資料であり、現行仕様は上記の主資料、検証可能性と保証statusは
`VALIDATION_STATUS.md`およびmachine-readable artifactを正本とする。

## 実装・検証資料

現行実装の保証範囲、検証状況およびAPI規約は、研究資料の閲覧順序とは分けて次を参照する。

- [検証状況](../../VALIDATION_STATUS.md)
- [事前検証カタログ実施証拠索引](prevalidation_catalog_evidence_map.md)
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

## M2 usable B2契約修正 v2（2026-10-04）

外部reviewの修正要求を[amendment v2](pr2_matched_accuracy_m2_transfer_contract_amendment_v2.md)へ反映した。
Pareto supportとprimary ratioは共にaccuracy-eligibleかつprimary重大underestimateのないB2だけを使う。
v1証拠・固定5構成・seed・196-wrapper上限を維持した。この契約修正時点では科学実行とheld-out accessは未認可だった。
moduleは`src/trotterlib/pr2_matched_accuracy_m2_transfer_contract.py`、runnerは
`scripts/run_pr2_matched_accuracy_m2_transfer_contract.py`、testは
`tests/test_pr2_matched_accuracy_m2_transfer_contract.py`、schema/planは
`artifacts/pr2_matched_accuracy_m2_transfer_contract/2026-10-04/`から辿れる。

## PR-2 M2科学実行コードの入口

[実装資料](pr2_matched_accuracy_m2_transfer_execution_implementation.md)は、科学実行module/runner/testと
契約v2正式planの関係、結果前identity、196-wrapper cap、全status後STOPを説明する。
local implementation testは科学transfer結果ではない。別authorizationと最終review前にheld-outを開かない。

## PR-2 M2最終実行前レビュー

- [実行authorization](pr2_matched_accuracy_m2_transfer_execution_authorization_v1.md)：source `2978e2f`、plan、候補/seed/compiler/fixed output、一回制限と全status後STOP。
- [最終review依頼](pr2_m2_execution_authorization_external_review_request_90a9f24.md)：authorization commit `90a9f24`をレビューする。承認と利用者の実行指示前はheld-outを開かない。
- [authorization監査](../../artifacts/pr2_matched_accuracy_m2_execution/2026-10-04/authorization_audit_v1.json)：128 source、196 keys、96 seeds、local focused84＋helper134 pass。M2科学結果ではない。

## 2026-10-12 N3理論・幅指向縮約taskの統合準備

文書入口：`docs/research/n3_theory_task_integration_v1_20261012.md`。
数学：`docs/research/n3_theory_claim_map_v1_20261012.md`、文献：`docs/research/n3_literature_correspondence_v1_20261012.md`、将来仕様：`docs/research/n3_width_reduction_v2_spec_20261012.md`。
保存帰結checker：`src/trottertracks/representation_exploration/n3_theory_implications_v1.py`、CLI：`scripts/verify_n3_theory_implications_v1.py`、合成tests：`tests/test_n3_theory_implications_v1.py`。
証拠・review/identity：`artifacts/n3_theory_task_integration_v1/2026-10-12/`。
30 synthetic PASS／保存全幅補題二例を独立照合。新science0、幅generator未実装、次はGPT研究判断、Mandatory STOP。

N3統合資料のGitHub取得確認：`docs/research/n3_theory_task_integration_v1_publication_20261012.md`（payload 2de3b26f、66 blobs、一致PASS、科学STOP）。

## N3幅比較 v2：比較実装・未使用入力の科学実行準備

文書入口 `docs/research/n3_width_comparison_v2_preparation_20261012.md`、数学 `n3_width_comparison_v2_contract_20261012.md`、結果前設計 `n3_width_comparison_v2_preregistration_20261012.md`、文献 `n3_width_comparison_v2_literature_amendment_20261012.md`（いずれもdocs/research）。
module群 `src/trottertracks/representation_exploration/n3_width_{contract,algebra,generator,check,pipeline,execution}_v2.py`。
runner `scripts/run_n3_width_comparison_v2.py`、保存verifier `scripts/verify_n3_width_comparison_v2.py`、入力literal `scripts/prepare_n3_width_inputs_v2.py`、tests `tests/test_n3_width_comparison_v2.py`。
artifacts `artifacts/n3_width_comparison_v2/2026-10-12/preparation/`。32 synthetic PASS、科学入力の数値評価0、実行認可待ち・Mandatory STOP。

N3幅比較v2：[固定plan](n3_width_comparison_v2_execution_plan_20261012.json)。最終source3eba0814、synthetic32・保存90証拠、registered数値評価0、明示実行認可待ち。

N3幅比較v2：[GitHub公開・再取得確認](n3_width_comparison_v2_publication_20261012.md)。payload9da12e27、52 blobs一致、remote preflight・保存合成90証拠PASS。科学実行0・明示認可待ち。

2026-10-13 N3幅比較v2：[固定pilot実行認可](n3_width_comparison_v2_execution_authorization_20261013.md)。6H、source/input/plan不変、attempt1/retry0、結果・監査公開後STOP。

N3幅比較v2の[結果公開・GitHub再取得](n3_width_comparison_v2_results_publication_20261013.md)：payload6d8f769b、59 blobs一致、remote独立1224証拠/36反証と4集計file一致。認可消費済み・GPT科学レビュー開始承認待ち・Mandatory STOP。

N3保存帰属v1の[固定plan](n3_saved_attribution_v1_execution_plan_20261013.json)：source a7a9f42d、9 source blob/40保存files identity PASS、Python3.12.3、新数値事後評価・PSD再認証の明示認可前。

N3保存帰属v1：[公開・remote取得確認](n3_saved_attribution_v1_publication_20261013.md)。payload08ffe034、81 blobs一致、identityと3 literal files byte一致。新数値事後評価・PSD再認証の明示認可前、Mandatory STOP。

- 2026-10-13 N3保存帰属v1 [一回実行認可](n3_saved_attribution_v1_execution_authorization_20261013.md)：固定保存データ事後評価＋3証拠再認証、結果後STOP。

- 2026-10-13 N3保存帰属v1 [結果・限界](n3_saved_attribution_v1_results_20261013.md)、[GPT判断引継ぎ](n3_saved_attribution_v1_results_review_handoff_20261013.md)。保存データ限定の独立監査 `scripts/audit_n3_saved_attribution_v1_results.py`、artifact `artifacts/n3_saved_attribution_v1/2026-10-13/evaluation1{,_audit}`。新producer/solverなし、STOP。

- 2026-10-13 N3保存帰属v1 [結果公開・remote保存照合](n3_saved_attribution_v1_results_publication_20261013.md)：85blob一致、取得saved audit byte一致、Mandatory STOP。

## 2026-10-13 量子化学relative blockの事前検証

[構成・方法差・誤差・費用](qc_relative_block_prevalidation_v1_20261013.md)、[科学判断引継ぎ](qc_relative_block_review_handoff_v1_20261013.md)。module `src/trottertracks/representation_exploration/qc_relative_block_v1.py`、synthetic runner `scripts/check_qc_relative_block_synthetic_v1.py`、tests `tests/test_qc_relative_block_v1.py`、artifacts `artifacts/qc_relative_block_prevalidation/2026-10-13/`。13 fixed synthetic PASS、実分子/eig/compile0。可逆関数回路とmethod delta/具体費用は未確定。pilot未認可・Mandatory STOP、旧Track/N3 source/results保全。

## 2026-10-13 相対時間発展block追加技術監査

[方法・費用監査](qc_relative_block_method_cost_audit_v1_20261013.md)、[GPT判断用資料](qc_relative_block_method_cost_handoff_v1_20261013.md)。artifacts `artifacts/qc_relative_block_method_cost_audit/2026-10-13/`。基準25cbc6e7のsource/test/runnerと保存13件は変更・再実行なし。既知primitiveのread-only調査と静的数式/資源会計、function費用UNKNOWN、独立差未特定。新科学計算・compile・synthetic0、詳細backend Phase2未起動、GPT判断前Mandatory STOP。数値validation pathは追加しない。

## 2026-10-13 chemistry mixed-access feasibility計画

[最小計画](qc_mixed_access_feasibility_plan_v1_20261013.md)、[次のscope承認](qc_mixed_access_feasibility_approval_scope_v1_20261013.md)。artifacts `artifacts/qc_mixed_access_feasibility/2026-10-13/`。current constructor独立算法非採択をユーザー経由の完了GPT判断として反映。新数値0、science module/runner/checkerは将来技術準備・source未freeze、旧13再実行なし。7法/H4一入力の有限PF/残余function費用を設計、launch認可なしMandatory STOP。数値validation実装pathを追加したとはしない。
