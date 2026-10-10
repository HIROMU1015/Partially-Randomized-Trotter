> 2026-10-07 Track B RA-D0 v3：**READY_FOR_RA_D0_ONE_SHOT_AUTHORIZATION**（実行承認ではない）。
> [source review](docs/tracks/algorithm_codesign/ra_d0_source_review_v3_20261007.md) / [GPT handoff](docs/tracks/algorithm_codesign/ra_d0_gpt_handoff_v3_20261007.md) / [manifest](artifacts/track_b_ra_d0_source_review_v3/2026-10-07/evidence_manifest_v3.json)。
> [focused verifier](scripts/tracks/algorithm_codesign/verify_ra_d0_source_review_v3.py) / [v3 tests](tests/tracks/algorithm_codesign/test_ra_d0_source_review_v3.py)。
> exact-certified B2 minimum infeasibilityを正常outcomeに修正。pointをfreezeへ記録しbudget/queryを空にして次nへ進む。
> uncertified failureはtechnical STOP。数値・candidate・grid・call/resource capは維持。登録最適化0、authorizationなし、mandatory STOP。

> 2026-10-07 Track B RA-D0 v2：**READY_FOR_SEPARATE_RA_D0_ONE_SHOT_REVIEW**。
> [source review](docs/tracks/algorithm_codesign/ra_d0_source_review_v2_20261007.md) / [GPT handoff](docs/tracks/algorithm_codesign/ra_d0_gpt_handoff_20261007.md) / [evidence manifest](artifacts/track_b_ra_d0_source_review_v2/2026-10-07/evidence_manifest_v2.json)。
> [future runner](scripts/tracks/algorithm_codesign/run_ra_d0_one_shot.py) / [focused verifier](scripts/tracks/algorithm_codesign/verify_ra_d0_source_review_v2.py) / [v2 tests](tests/tracks/algorithm_codesign/test_ra_d0_source_review_v2.py)。
> B0_saved/ideal分離、数値B1⊂B2⊂B3、profile-paired budget、batch freeze-before-B3、
> anchor-first、main LP 55,275 / auxiliary込み110,550、resource/launch guardを固定。
> 登録最適化・実budget/minimum/witness取得0、authorizationなし。旧本文・旧STOP・Track Aは保持。mandatory STOP。

## Track B RA-D0 static preparation / mandatory STOP（2026-10-06）

[source review](docs/tracks/algorithm_codesign/ra_d0_source_preparation_review_v1.md)：21 columns/x、18 sign pairs一致、35 focused tests PASS。
B専用namespace `src/trottertracks/algorithm_codesign/ra_d0/`、static generator／verifier、
`artifacts/track_b_ra_d0_preparation/2026-10-06/`に候補・grid・semantic audit・provenanceを保存。
ideal nestingと数値membershipを区別し、query実行scope／資源上限をGPTへ返す。
登録最適化／新合成／新science0、RUN_READY=false、既存分類・Track A・旧STOP保持。
**mandatory STOP。以下の既存本文を全文保持する。**

## Track B RA-RTE統合数学監査・mandatory STOP（2026-10-06）

R1.5 `af3d014d0a0cfcbbd25bb544f6544652fec92942` 基点、GPT設計案へのDOCS_SYMBOLIC_ONLY_MATHEMATICAL_AUDIT。
[命題別監査](docs/tracks/algorithm_codesign/ra_rte_mathematical_audit_v1.md)と[GPT handoff](docs/tracks/algorithm_codesign/ra_rte_mathematical_audit_gpt_handoff_20261006.md)：一block／finite table／canonicalのfixed-n LPは仮定付きで成立。
shot-gridの固定total cap保存には反例。log/root・sampler認証、Delta=0、peak workspaceの規約を実行前修正へ返す。
[stdlib人工bookkeeping](scripts/tracks/algorithm_codesign/check_ra_rte_mathematical_bookkeeping.py)の[50 checks](artifacts/track_b_ra_rte_mathematical_audit/2026-10-06/bookkeeping_checks_v1.json)を一般証明と分離した。
[manifest](artifacts/track_b_ra_rte_mathematical_audit/2026-10-06/evidence_manifest_v1.json)、[dated note](docs/research/研究ノート/2026-10-06_track_b_ra_rte_mathematical_audit.md)。science/synthesis/solver/資源再採点0、共通API変更0。
既存科学分類・結果・STOPは不変。性能・新規性・algorithm採択・次実装／R2 authorizationは未確定。
**mandatory STOP。次の採択・実装・pilotの必要性／範囲はGPT判断。以下の既存本文を全文保持する。**

## Track B R1.5保存値帰属・mandatory STOP（2026-10-06）

input R1 commit `24bfeb84a4ce87b56985d174dd98d1d5e1702a2b` の保存値だけを用いたPOSTHOC attribution / design input。
[帰属報告](docs/tracks/algorithm_codesign/r1p5_saved_value_attribution_v1.md)と[GPT handoff](docs/tracks/algorithm_codesign/r1p5_gpt_handoff_20261006.md)：新science/synthesis/compile/候補追加0、R1科学分類は不変。
primaryは2-qubit finite P₃、distinct-basis controlled、x={1/8,1/4}、登録native三precision。
Aは登録(G_T,G_CX,G_1Q) frontにx=1/8の1e-4、x=1/4の1e-3/1e-4で残る。
normalizationだけでなくnative費用とbias/shotの関係を整理し、固定合成列への依存も保存した。
[stdlib保存値解析](scripts/tracks/algorithm_codesign/analyze_r1p5_saved_attribution.py)、[全summary](artifacts/track_b_r1p5_saved_attribution/2026-10-06/attribution_summary_v1.json)、[provenance manifest](artifacts/track_b_r1p5_saved_attribution/2026-10-06/evidence_manifest_v1.json)、
[日付note](docs/research/研究ノート/2026-10-06_track_b_r1p5_saved_attribution.md)。共通library変更・独立validation・新algorithm採択はない。
限定診断SUPPORTS_RA_RTE_DESIGNは設計入力のみ。eta探索/R2/DF接続/追加scienceは未認可。
**mandatory STOP。次の数学設計・研究方針判断はGPT側。以下の既存本文を全文保持する。**

## Track B R1一回結果・mandatory STOP（2026-10-06）

固定S `d43d64a821a0249a0dfab12a2472bd3a72fdee74` →直接子authorization-only A
`411f08f768244fe87b600d82308c3851847fe9e4`からrun1/retry0。
[結果照合](docs/tracks/algorithm_codesign/r1_one_shot_result_validation_20261006.md)：126 keys / 264 rows / 132 controlled tasks完了、全task適格。
terminal R1_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW、[保存field専用監査](scripts/tracks/algorithm_codesign/audit_r1_saved_result.py) PASS。
primary distinct-basis controlledではB²改善とnative/shot資源のtrade-offを保存し、自動研究GOはない。
[全264 rows CSV](artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/resource_rows_display_v1.csv)、[evidence manifest](artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/evidence_manifest_v1.json)、
[GPT判断への入口](docs/tracks/algorithm_codesign/r1_post_run_gpt_review_request_20261006.md)。原result/marker/source、既存証拠・共通API・Track A保持。
science終了後mandatory STOP、追加合成/target/grid/分子/DF/trajectory/GPUは行わない。
研究方針・RQ・新規性・着地点・追加検証の必要性/範囲はGPT側。
以下のpending/最新記述は当時の履歴として本文をそのまま保持する。

# プロジェクト案内

## Track B R1 source preparation — source review / mandatory STOP

2026-10-06の[preregistration v2](docs/tracks/algorithm_codesign/rte_reallocation_r1_preregistration_v2.md)、
[native semantics](docs/tracks/algorithm_codesign/rte_reallocation_r1_native_semantics_v1.md)、
[GPT source review](docs/tracks/algorithm_codesign/rte_reallocation_r1_source_review_request_20261006.md)。
ordinary/PTSC-K0/A、PauliだけCTS control、distinct-basis controlled primary。canonical/resource vector。
[B専用source](src/trottertracks/algorithm_codesign/rte_reallocation/)、
[runner](scripts/tracks/algorithm_codesign/run_r1_rte_reallocation.py)、
[semantic tests](tests/tracks/algorithm_codesign/test_r1_rte_reallocation_semantics.py)、
[accounting/launch tests](tests/tracks/algorithm_codesign/test_r1_rte_reallocation_accounting_launch.py)、
[source manifest](artifacts/track_b_rte_reallocation_r1_source/2026-10-06/source_manifest_v1.json)。
27 focused tests PASS、42 static angles/126 keys、264予定rows。登録synthesis/resource取得0、R1未認可。
旧R0/R0.5/科学証拠・共通API・Track A保持。以下の本文は履歴を含めそのまま保存。


## Track B R0.5: scoped equivalence audit / GPT review

2026-10-06のdocs/symbolic-only [監査](docs/tracks/algorithm_codesign/rte_reallocation_r05_equivalence_novelty_audit_v1.md)と
[GPT handoff](docs/tracks/algorithm_codesign/rte_reallocation_r05_gpt_handoff_20261006.md)を追加。
指定P0–P3内の限定classification `METHOD_DELTA_CANDIDATE`、gate `CONDITIONAL-R1`。
R1の必要性はGPT判断、実行は未認可。固定9条件/18 signのtechnical比較のみ一回。
[checker](scripts/tracks/algorithm_codesign/check_rte_reallocation_r05_symbolic.py)、
[manifest](artifacts/track_b_rte_reallocation_r05/2026-10-06/audit_manifest_v1.json)、
[dated note](docs/research/研究ノート/2026-10-06_track_b_rte_reallocation_r05.md)を参照。
旧R0/科学証拠・Track Aは保持し、下記の旧本文はそのまま残す。mandatory STOP。


## Track B R0: finite-mean RTE representation audit (2026-10-06)

- [Review packet](docs/tracks/algorithm_codesign/rte_reallocation_r0_review_packet_20261006.md), independent all-odd proof, scoped prior art and conditional R1 proposal.
- [Technical checker](scripts/tracks/algorithm_codesign/check_rte_reallocation_symbolic.py) and [R0 provenance](artifacts/track_b_rte_reallocation_r0/2026-10-06/audit_manifest_v1.json). A80/B9 exact fixtures; one technical run, science 0.
- R0 math checked; new-method priority/resource value unresolved. Existing STOPs remain; RUN_READY=false; GPT review required.


Track B最新は[BS-0.5 method/target設計監査](docs/tracks/algorithm_codesign/bs05_method_target_design_audit_v1.md)。
現candidateは同じsparse/operator LCUと別delta未定義のため専用arm/new-method claimを外す。
ordinary finite-RTEの形式仕様、O/C別比較、多資源Pareto、Wada grouped LCUをreviewへ戻す。
12 targetはcommuting control4＋performance8。新実装/science/tests0、RUN_READY=false、mandatory STOP。
[監査manifest](artifacts/track_b_bs05_design_audit/2026-10-06/audit_manifest_v1.json)。application採否・route closureはGPT判断。
旧design v1、SP results/markers、BF/BM closure、A証拠は維持。以下の「最新」は判断履歴。

Track B最新は[SP-1後GPT reviewのblock合成仕様化](docs/tracks/algorithm_codesign/block_synthesis_design_review_20261006.md)。
数学/API、claim/強い対照、小型pilot未承認案を一つの入口へ整理したdocs-only準備。
有限精度coherent対象・block単位・誤差配分を比較するRQ案。finite-RTE平均Mの直接合成は未実証候補。
[input identity／設計manifest](artifacts/track_b_block_synthesis_design/2026-10-06/)。実装/新science/tests0、RUN_READY=false。
既存SP-1 result/marker・BF/BM closure・A証拠を保持。資料公開後STOPし、具体domain/辞書/対照/会計の採否はGPTへ戻す。
以下の「最新」は当時の履歴。

Track B最新は[SP-1一回結果・保存値監査](docs/tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md)。
固定S `0d01ed9a`→direct authorization-only A `9630e061`でrun1、SP1_RESOURCE_MAP_COMPLETE_AWAITING_REVIEW。
48 rows／96 axes完了、42 rows適格／6 rowsモデルshot cap。technical failureなし、保存値監査PASS。
CのDRはn=8/16でgain、32でloss、64でcap。CのD-only/R-only material gainは0。
[結果・marker・evidence manifest](artifacts/track_b_sp1_wrapper_result/2026-10-06/v1/)、
[保存field専用監査](scripts/tracks/algorithm_codesign/audit_sp1_saved_result.py)、
[GPT側の研究再評価依頼](docs/tracks/algorithm_codesign/sp1_post_run_gpt_review_request_20261006.md)。
mandatory STOP、retry0、追加science未認可。以下のsource準備・未実行statusは当時の履歴として保持する。

Track B最新は[SP-1結果前契約・source最終review](docs/tracks/algorithm_codesign/sp1_wrapper_preregistration_v1.md)。
GPTがdomainを採用し、必須修正を反映した。role/mask非依存pre-placement fusion、加法的primitive T費用。
16登録pathの静的監査はfusion candidate/actual/cross-roleすべて0。実adapter/runnerと59 focused testsを追加。
[source manifest](artifacts/track_b_sp1_wrapper_source/2026-10-06/source_manifest_v1.json)。
science sweep/合成/trajectory/GPUは0、RUN_READY=false、authorizationはpending。
新Sを固定してGitHub公開後、最終source reviewへSTOP。以下の「未完了」は受領前の履歴。

Track Bの最新準備は[SP-1 wrapper累積・placement契約案](docs/tracks/algorithm_codesign/sp1_wrapper_preregistration_proposal_v1.md)。
GPT reviewを具体化した12 synthetic wrapper／4 mask案と、独立namespaceのFraction会計kernel／17 focused tests。
science sweep・合成・trajectoryは0、RUN_READY=false。common fusion／実adapter／source freeze／別authorizationは未完了。
[準備manifest](artifacts/track_b_sp1_wrapper_preparation/2026-10-06/preparation_manifest_v1.json)。
SP-0.5結果・marker、BF/BM closures、A証拠を保持し、公開後STOPして契約案reviewへ戻す。
以下は受領前の履歴。

Track B最新は[SP-0.5一回実行・保存値監査・GPT handoff](docs/tracks/algorithm_codesign/sp05_one_shot_result_validation_20261006.md)。
source `65f6fcdb`、直下authorization `9477cd2f`のHEADで一回だけ実行し、PRIMITIVE_TRADEOFF_EXISTS。
23/23 keys・16/16 rows、strict14、zero-cost1、J=1 control1、error/cap/runtime/numeric failureなし。
[元resultとmarker／監査manifest](artifacts/track_b_sp05_economics_result/2026-10-06/v1/)、
[保存field専用監査script](scripts/tracks/algorithm_codesign/audit_sp05_saved_result.py)。J再評価・science rerunなし。
mandatory STOP、retry0、wrapper pilot未認可。cheap exact notchのprimitive確認のみ、研究方針はGPTへ戻す。
以下の準備・未実行の記述は結果前履歴として保持する。


Track Bの現段階は[SP-0.5経済性gate・結果前source review](docs/tracks/algorithm_codesign/sp05_synthesis_economics_preregistration_v1.md)。
GPT reviewに従い16-cell wrapper pilotの前に、合成器一つ・exact π/4 catalogue一つ・8 target／23 keysを固定する。
B専用実装と34 focused technical testsは準備済み。登録target合成・J採点0、RUN_READY=false、別実行承認待ち。
[preparation manifest](artifacts/track_b_sp05_economics_preparation/2026-10-06/preparation_manifest_v1.json)。
B-F限定closure／BM現new-method closure／旧BM-1未実行／過去STOPを保持する。以下の「最新」は受領前の履歴。


Track B最新は[BM-0.5後の合成・placement設計仕様案](docs/tracks/algorithm_codesign/synthesis_placement_design_review_20261006.md)。
GPT reviewに従いB-F限定仮説とB-M現adapterのnew-method路線を閉じ、旧BM-1は未実行のまま保持する。
新候補の四mask、ancilla込みPAI、有限bias／weight／shot／T会計、原始pilot案を一文書と小さいJSONへ整理した。
一般原理は既知、利益・新規性は未実証。angle inventoryは確認した保存記録では未確立。
科学実行・実装・testsは0、次段未認可、利用者/GPT reviewへSTOP。以下のBM-0.5以前は履歴として読む。

Track B最新は[BM-0.5同値性監査](docs/tracks/algorithm_codesign/bm05_review_packet_20261005.md)。
固定nested列のcompact BCHとBM分解は三次で同値。同backend／同集約ならscoreも一致、現new-method gateは不通過。
BM-1未実行・未認可。applicationの必要性／別deltaはGPTへ返してmandatory STOP。以下はBM-0までの履歴。

Track Bの最新文書は[BF-A後review・BM-0 packet](docs/tracks/algorithm_codesign/bm0_review_packet_20261005.md)。
GPT reviewのB-F限定closureを記録し、B-M native列・DF評価・小型pilotを具体化した設計案。
新scienceは0、B-M新規性／BM-1 scopeはreview待ち、実行未認可、mandatory STOP。以下は既存履歴を保持する。

GPTによる研究Bの全面再設計は[固定資料handoff index](docs/tracks/algorithm_codesign/research_redesign_handoff_20261005.md)から読む。
BF-A・過去STOP・最新Aの別commit参照・未公開既存文書snapshotをまとめたdocs-only入口。新科学計算・方針採択なし、mandatory STOP。

Track B worktreeの現在の入口は[Algorithm Co-design](docs/tracks/algorithm_codesign/README.md)。
最新は[BF1-R0復元結果](docs/tracks/algorithm_codesign/bf1_read_only_recovery_result_validation_20261005.md)：
read-only recovery complete、事後にpreregistered primary BF-Aを復元。全結果後mandatory STOP、研究判断はGPT側。
新たな利用者指示は[BF1-R0 read-only recovery](docs/tracks/algorithm_codesign/bf1_read_only_recovery_contract_v1.md)。
既存cellからの一回の復元だけを認可し、science rerunは認可しない。module/runner/tests/contractはB専用namespace。
原science実行は[BF-1一回実行の結果照合](docs/tracks/algorithm_codesign/bf1_one_shot_result_validation_20261005.md)。
別branchの[JSON保存修正と限定検証](docs/tracks/algorithm_codesign/bf1_serialization_repair_review_20261005.md)は
技術作業のみ。研究方針全体の修正はGPT側で扱い、科学再実行は認可しない。
JSON保存例外により`INCONCLUSIVE`、mandatory STOP、retryなし。B code/runner/test/preparation/result artifactは
同READMEから辿る。以下のA記録はこのbranchのM2 result snapshotであり、並行Aの最新状態へは更新しない。

最終更新：2026-10-04

このファイルは、人またはGPTがリポジトリ全体を読むときの入口である。研究内容の正本、
実装、検証コード、結果データ、発表資料を区別し、古い研究経路を現在の結論として読まない
ための案内をまとめる。

PR-2の最新状態は、旧S0 STOPとdevelopment-only S2結果を保持し、matched-accuracy resource-map研究の
compile-free M1-Aを`SELECTION_LIMITED`で停止した後、`PROCEED_BOUNDED_COMPILE_EXPANSION`として
M1-B1を実行・検証した段階である。旧16-cell selectorは「proxyではactual frontierを完全に保持できなかった」
監査結果として保存する。M1-Aでaccuracy適格だったB2/B3 194 cellを32 trajectory・Re/Im二軸、B0/B1 16 cellを
二軸で測る12,448-wrapper mapは全件完了した。actual six-metric ParetoはB2 rank 3、q=1の2件で、
primary RZ point minimumは`B2-rank3-q1-r4-K2`、状態準備cost感度のlower envelopeも全てB2だった。
研究判断は`CONTINUE_RESOURCE_STUDY`。その後、development actual Pareto 2件とB0/B1/B3代表を合わせた
5構成、primary RZ、6指標Pareto、10% materiality、重大cost underestimate、4 terminal statusを
M2 held-out transfer契約へzero-compute固定した。入口は
`docs/research/pr2_matched_accuracy_m1_b1_bounded_compile_contract_v1.md`と
`docs/research/pr2_matched_accuracy_m1_b1_execution_contract_amendment_v2.md`と
`docs/research/pr2_matched_accuracy_m1_b1_execution_authorization_v1.md`と
`docs/pr2_matched_accuracy_m1_b1_result_validation.md`と
`docs/research/pr2_matched_accuracy_m2_held_out_transfer_contract_v1.md`。追加96 trajectory、held-out H4 1.30 Å、
transfer、S3は当時未実行・未承認だった。M2外部reviewの修正要求は
`docs/research/pr2_matched_accuracy_m2_transfer_contract_amendment_v2.md`へ反映し、
usable B2だけをsupportとratioに使うv2 source/planをcommit固定した。science実装の入口は
`docs/research/pr2_matched_accuracy_m2_transfer_execution_implementation.md`である。
actual source `2978e2f`、別authorization `90a9f24`、最終review承認と利用者指示を経て、M2を一度だけ実行した。
最新は[結果照合](docs/pr2_matched_accuracy_m2_transfer_result_validation.md)：H4 1.30 Å、STO-3G、DF rank12、
`T=0.8`の固定5構成、196 wrapperが完了し、B2二件がParetoに残って`TRANSFER_SUPPORTED`。
result commitへ収録するlocal execution evidenceで、固定構成のtransfer以外へ一般化しない。現在はmandatory STOP、研究方針全面review待ち。
追加96、held-out再探索、S3、別geometry/分子へ進まない。

## 最初に読む順序

1. [`docs/research/研究概要・現状.md`](docs/research/研究概要・現状.md)
   研究目的、採用中の前提、現在の実装・検証段階、主要結果、未解決事項の最新要約。
2. [`docs/research/研究目的・研究課題.md`](docs/research/研究目的・研究課題.md) と
   [`docs/research/研究方法・解析手順.md`](docs/research/研究方法・解析手順.md)
   研究設計と解析手順の正本。
3. [`VALIDATION_STATUS.md`](VALIDATION_STATUS.md)
   各結果を再利用できるか、欠落データや再計算が必要かを示す状態表。
4. [`artifacts/validation_manifest.json`](artifacts/validation_manifest.json)
   検証結果、生成コード、テスト、成果物を結ぶ機械可読な証拠台帳。
5. [`docs/research/prevalidation_catalog_evidence_map.md`](docs/research/prevalidation_catalog_evidence_map.md)
   事前検証カタログのうち、実施したID・work packageと文書、artifact、runner、testの対応表。
6. 数値を使う場合だけ、対応する `docs/*_validation.md` と `artifacts/` のJSONを確認する。

`docs/research/研究ノート/` は意思決定の時系列記録であり、現在の仕様ではない。過去の暫定値が
後日のノートや研究概要で変更されている場合は、最新の研究概要と規範文書を優先する。

## 現在の研究段階

2026-09-25現在、A0後のP-B/P-C/P-Aテーマ選定と、各候補の停止点まで完了した。P-Bは現H4 gridで
実用的signal差がなく停止した。P-A v1はblind transferを通過したが、形式化とforced-support
Taylor-order-2検証でinterval分割・一区間baselineとの差が全30 taskで0となり、interval claimを停止した。
P-Cは0.80--1.20 Åの局所pilotを通過したが、事前登録した0.70--1.60 Å tracking・breakdown検証では
追跡prefixが独立prefixと全点同一で、1.40/1.60 Åのcoefficient予測誤差が33.653%/123.245%となった。
pair予測、continuity診断、mechanism discriminationも固定gateを通らず、
`stop_pc_current_h4_family_as_primary`となった。

従って、A/B/Cに確認済みの主研究候補はない。P-Dのpilotと現実化gateは通過したが、S0/S1の
公平再最適化ではB1b/B2/B4が全scopeで同じnew fourth、`delta=0.2,R=16`を選び、B2のB4 regretは0、
限定K4でも選択は変わらなかった。finite補正が選択を変えるCase C/Dの証拠は得られていない。
outer-stageだけのB1aは一段延長後も`m_D=128`上限へ達したため、一次分類Case Bに
`undetermined_boundary`を付け、GO判定を出さず停止した。P-Dの研究方針とbaseline設計を再検討し、
H12、長RPE、compiled総costへはまだ広げない。
その後、固定S1 artifactの事後再解析で主baselineのB1b/B2/B4一致を確認し、P-D S2を停止した。
R3も一般multi-fidelity法との差分を固定できず`STOP_R3_NO_METHOD_DELTA`となった。2026-09-26には、
有限RTE打切り誤差をHadamard複素信号の位相方向と半径方向へ分ける新候補についてFR-0を完了し、
事前登録済みFR-1を固定2×2 toyで実行した。補正後演算子$A_{\rm corr}$と実際の平均
$A_{\rm mean}=A_{\rm corr}/\mathcal B$を区別した境界は495適用recordで違反0、semantic・負時間・K4・
非対称配置も通過した。一方、利用可能な$\underline\rho=0.8$でのG2は不通過で、真の$\rho$を使う
場合だけ改善した。その後のFR-R1aでは正scalar処理後の片側差が0、FR-R1bでは同情報のstrict gainが
8件あったが固定予算の片側認証差が0となり、現行判断は`MECHANISM_ONLY_NO_PRACTICAL_GO`である。
FR-R2は開始せず、2026-09-27に研究を候補探索から理論成果の完成へ切り替えた。C1/C2を中核、C3を
条件付き応用とし、先行研究との定理単位の照合と証明義務T1--T4を終えるまで新しい数値計算を行わない。

それ以前の中心課題は、DF Hamiltonianを決定論部分とランダム部分へ分けたpartial-$S_2$について、有限RTE、
RPEの信号半径・測定回数、1 shot当たりのコンパイル後回路コストを接続することだった。PF係数、有限RTE、
ランダム回路コスト、短いRPE段の接続は限定条件で検証済みだが、最終総cost評価には達していない。
既存設定`CA/10`を暫定目標にすると$q_{\max}=32768$が必要で、従来の固定
$\delta=0.1,r=4,K=2$は長roundへ単純外挿できないことを確認した。その後、H4の
実行済み$\delta$窓でround別$(r_m,K_m)$を再探索し、$\delta=0.01,0.0125,0.02$に
行列検査を通るscheduleを構成した。さらに、短時間幅0.02--0.000390625で局所回路指標が
変わらないことと、イベント列長8、16、32への移送を検査し、$r\leq16$では1--3イベント、
$r=32$では4イベント補正を用いる中央RTEブロックcost proxyを接続した。この限定proxyでは
$\delta=0.02$が全6指標で最小となった。

2026-09-21に、広い検証backlogを一括実行せず、研究方向を選ぶGate S1を先に置く方針を採用し、
最初の`WP00 -> WP02 -> WP01-S`を実行した。H4 rank-12固定snapshotへPF入力を結び直し、
CA/CA/10/CA/100のround horizonを監査した。CA/10は既存3 schedule・56点行列検査を再利用できるが、
CA/100は既存3つの$\delta$がすべて経験的PF予算を超える。CA/10の条件付き比較では
$L_D=0$を拡張した解析的成分作用数proxyでscreen outし、$L_D=12$の点推定は$L_D=3$より
21.0%低かった。
ただし保守的な長$q$移送scenarioでは区間が重なるため、結論は未決定である。続くWP04では、
両候補に公平な$\beta$・$\alpha$再配分を与えると決定論endpointの点推定差は4.96%へ縮み、
5%・25%区間がともに重なることを確認した。共通の主要因は$\beta$、次いで$\alpha$再配分で、
$L_D=3$のcompiled-cost整合round schedule単独の利得は固定schedule比1.93%だった。WP03では
$C_D$、論文D6、支配固有位相係数だけを差し替えた18条件の選択が全て
$L_D=12,\delta=0.02$で変わらず、係数選択も区間重なりを解消しなかった。Gate S1では
これらを統合し、区間判定を「同点」でなく`undetermined`、最大の残存不確かさをfull controlled
interrogationの回路scope・構造とした。T4とT7を主軸、T1/T2/T5/T6を限定継続、T3を保留とし、
次の一件をWP06-aとした。WP06-aではsupport限定Gaussian completionが単一Z/ZZ eventのRZを
39--62%減らした一方、異なるsupportの長さ3列ではfull basis共有より15.0%増え、事前の5% triggerが
発火した。control・relative-phaseの現行方針は同値性を通過した。WP06-bでは独立trainingから、同一
元basisのsingleton runだけsupport限定へ置換するpolicyを固定した。未使用列長3, 6でRZ -10.67%、
CX -8.13%、total depth -2.25%、最大operator残差$1.34\times10^{-15}$だった。中央RTE差だけを既存
proxyの$q$ slopeへ加えたbridgeでは$L_D=3/12$の点順位が反転した。続くWP05-aでは選択policyを
complete controlled partial-$S_2$／Hadamard wrapperへ接続し、$q=1,2$較正から未使用$q=4$をRZ最大
2.29%で予測した。中央additive bridgeのfull-wrapper RZ残差も最大2.63%で5%基準を通過した。
固定WP04条件の点順位は$L_D=3$となったが区間は重なった。WP05-bでは$q=8$と比較対照
$\delta=0.01$へ拡張し、$\delta=0.02,r=32,q=8$の初回5%逸脱を独立32 trajectoryで再検証した。
再検証では選択policyのRZ誤差0.52%、全metric最大0.54%、full basisのRZ誤差0.83%となり、
5%基準を通過した。続くWP01-D/C07では$\alpha$・shot数を候補ごとに再最適化し、点推定で
$L_D=3$が$L_D=12$より13.92%低かった。5% local model区間は僅かに分離した一方、25%移送区間は
重なるため、頑健な方向判断は未確定である。G08で後半3 roundへのcost集中を確認し、M08の
$q=16,32$直接holdoutはselected RZ 2.466%、観測RZ最大3.286%で通過した。これらの実測幅による
再集計ではlocal区間が分離するが、直接domainは$q\leq32$で25%移送区間は重なるため、頑健判定は
変わらず、現比較は最終的な科学的優位性評価ではない。
2026-09-23のM06/L08では、同一trajectoryをoptimization level 2で再compileした。q=16,32 proxyは5%基準を通過したが、固定plan focused再集計の点推定差は8.42%、区間分離上限は1.881%となり、実測selected RZ discrepancy 2.340%で区間が重なった。従ってcompilerをまたぐlocal分離は未確立で、頑健判定は`undetermined_under_compiler_and_transfer_sensitivity`である。
続くN07/P03では不確かさをsampling、model bias、compiler、長q移送、状態準備、外部移送に分離し、状態準備をRZ相当/shotのパラメータとして再集計した。`L_D=3`は2,376 shot多く、共通準備costは常に点推定利得を縮める。点推定break-evenはopt1で約9,905万、opt2 focusedで約4,707万RZ相当/shotだが、opt2 focusedはP=0ですでに区間が重なり、compiler-robustな区間優位性は確立しない。WP11では11個のartifactをT1--T7へ統合し、T4/T7を主軸、T1を範囲変更、T3を保留とした。次の一件は`L_D=3`のopt2未測定`r=1,2,4,8,16`を埋めるall-r coherent opt2再最適化であり、外部instance pilotは棄却せずその後へ延期する。
固定条件、数値、成果物は
[`research_direction_prevalidation.md`](docs/research_direction_prevalidation.md)と
[`research_direction_ablation.md`](docs/research_direction_ablation.md)、
[`research_direction_pf_sensitivity.md`](docs/research_direction_pf_sensitivity.md)、
[`research_direction_gate_s1.md`](docs/research_direction_gate_s1.md)、
[`research_direction_structure_pilot.md`](docs/research_direction_structure_pilot.md)、
[`research_direction_sequence_policy.md`](docs/research_direction_sequence_policy.md)、
[`research_direction_full_scope.md`](docs/research_direction_full_scope.md)、
[`research_direction_full_scope_extension.md`](docs/research_direction_full_scope_extension.md)、
[`research_direction_decision_cost.md`](docs/research_direction_decision_cost.md)、
[`research_direction_late_round_proxy.md`](docs/research_direction_late_round_proxy.md)、
[`research_direction_compiler_transfer.md`](docs/research_direction_compiler_transfer.md)、
[`research_direction_uncertainty_break_even.md`](docs/research_direction_uncertainty_break_even.md)、
[`research_direction_wp11_synthesis.md`](docs/research_direction_wp11_synthesis.md)に記録する。

最終的な全RPE段の総コスト最適化と、決定論PFに対する最終的な優位性評価はまだ行っていない。
最新の到達点と次の検証は、必ず
[`研究概要・現状.md`](docs/research/研究概要・現状.md)で確認する。

## ディレクトリの役割

| 場所 | 役割 | 読み方 |
|---|---|---|
| `src/trotterlib/` | ライブラリ本体 | 現行実装と検証ロジック。分類は[`src/trotterlib/README.md`](src/trotterlib/README.md) |
| `scripts/` | 実行入口 | 検証runner、batch、診断、旧経路。分類は[`scripts/README.md`](scripts/README.md) |
| `tests/` | 自動テスト | API・数値恒等式・成果物schemaの回帰検査。科学的結論そのものではない |
| `docs/research/` | 研究方針の正本 | 概要、目的、方法、評価計画、研究ノート |
| `docs/` | 実装・検証の説明 | 各検証の条件、結果、限界、実装規約。索引は[`docs/README.md`](docs/README.md) |
| `artifacts/` | 計算結果と入力snapshot | JSON等の証拠、キャッシュ、途中状態。利用規則は[`artifacts/README.md`](artifacts/README.md) |
| [`partial_randomized_trotter_prevalidation_catalog.md`](partial_randomized_trotter_prevalidation_catalog.md) | 研究方向を選ぶための事前検証backlog | 実行層・Gate S1・条件付きbranchの索引。実施済み範囲との対応は[`prevalidation_catalog_evidence_map.md`](docs/research/prevalidation_catalog_evidence_map.md)を参照 |
| `BentoSlide構成案*.md` | 発表資料生成用の指示 | 研究の正本ではない。位置づけは[`docs/presentations/README.md`](docs/presentations/README.md) |
| ルートのPDF | 参考論文または発表資料 | 種別は[`docs/references/README.md`](docs/references/README.md)で確認 |

## 実装と検証の関係

```text
研究方針・条件
  docs/research/
        ↓
ライブラリ実装
  src/trotterlib/
        ↓
実行入口                 回帰テスト
  scripts/run_*.py   ↔    tests/test_*.py
        ↓
結果・入力snapshot
  artifacts/
        ↓
結果の説明と状態
  docs/*_validation.md
  VALIDATION_STATUS.md
  artifacts/validation_manifest.json
```

`scripts/run_*.py` はメインロジックの置き場所ではなく、引数を受け取り
`src/trotterlib/` の処理を呼び出して成果物を保存する入口である。検証名と同名の
ライブラリ、runner、test、文書、artifactを一組として読む。

## 共有サーバー向けexecution infrastructure

長時間検証を独立taskへ分割して安全に実行・再開する経路は、
`src/trotterlib/parallel_validation_executor.py`、`scripts/run_parallel_validation_batch.py`、
`tests/test_parallel_validation_executor.py`、
[`docs/server_parallel_validation_execution.md`](docs/server_parallel_validation_execution.md)を一組として読む。
これは実装・運用基盤であり、新しい科学的検証結果や最終総cost評価ではない。

WP11が選択したM06-F計算経路は、`src/trotterlib/research_direction_full_opt2.py`、
`src/trotterlib/research_direction_full_opt2_completion.py`、
`src/trotterlib/research_direction_full_opt2_extension_analysis.py`、
`src/trotterlib/research_direction_proxy_lineage_reconciliation.py`、
`scripts/run_research_direction_full_opt2_compute.py`、
`scripts/run_research_direction_full_opt2_analysis.py`、
`scripts/run_research_direction_full_opt2_completion.py`、
`scripts/run_research_direction_full_opt2_extension_analysis.py`、
`scripts/run_research_direction_proxy_lineage_reconciliation.py`、
`tests/test_research_direction_full_opt2.py`、
`tests/test_research_direction_full_opt2_completion.py`、
`tests/test_research_direction_full_opt2_extension_analysis.py`、
`tests/test_research_direction_proxy_lineage_reconciliation.py`、
[`docs/research_direction_full_opt2.md`](docs/research_direction_full_opt2.md)を一組として読む。
初期36 taskとfresh-32拡張15 taskは51/51で完了し、両gate通過後のcoherent再最適化も完了した。
A0は新規compileなしで最新fresh `q=1,2` proxyを旧`q=16,32`固定holdoutへ再照合した。
compute resultは`artifacts/research_direction_full_opt2/2026-09-24/`、最終監査・解析は
`artifacts/research_direction_full_opt2/2026-09-25/`に置く。

2026-09-25からはA0完了後の研究テーマ選定を`P-B -> P-C -> P-A`で行う。P-Bは
`src/trotterlib/research_direction_signal_weight_pilot.py`、
`scripts/run_research_direction_signal_weight_pilot.py`、
`tests/test_research_direction_signal_weight_pilot.py`、
[`docs/research_direction_signal_weight_pilot.md`](docs/research_direction_signal_weight_pilot.md)を一組として読む。
現H4 gridでは案Bを進めるsignal差が得られず停止した。P-Cは
`src/trotterlib/research_direction_geometry_energy_difference_pilot.py`、
`scripts/run_research_direction_geometry_energy_difference_pilot.py`、
`tests/test_research_direction_geometry_energy_difference_pilot.py`、
[`docs/research_direction_geometry_energy_difference_pilot.md`](docs/research_direction_geometry_energy_difference_pilot.md)、
`artifacts/research_direction_geometry_energy_difference_pilot/2026-09-25/`を一組として読む。
固定H4 5 geometryの未使用geometry/delta差分bias予測gateを通過して案Cを候補として残し、
P-Aは`src/trotterlib/research_direction_joint_synthesis_pilot.py`、
`scripts/run_research_direction_joint_synthesis_pilot.py`、
`tests/test_research_direction_joint_synthesis_pilot.py`、
[`docs/research_direction_joint_synthesis_pilot.md`](docs/research_direction_joint_synthesis_pilot.md)を一組として読む。
未使用列長3、5、8でinterval-union DPが現行policy比pooled RZを7.19%減らし、全gateを通過した。
3 pilotの比較は`research_direction_theme_selection.py`と同名runner/test、
[`docs/research_direction_theme_selection.md`](docs/research_direction_theme_selection.md)へ固定した。
P-Aを暫定主題、P-Cを副候補、P-Bを現範囲で停止とした。続くscoped prior-art auditは
[`docs/research/pa_joint_synthesis_prior_art_audit.md`](docs/research/pa_joint_synthesis_prior_art_audit.md)に、
計算前のblind条件は
[`docs/research/pa_joint_synthesis_blind_validation_preregistration.md`](docs/research/pa_joint_synthesis_blind_validation_preregistration.md)に固定した。
実装は`src/trotterlib/research_direction_joint_synthesis_blind_validation.py`、
`scripts/run_research_direction_joint_synthesis_blind_validation.py`、
`tests/test_research_direction_joint_synthesis_blind_validation.py`を一組として読む。compile前dry-runで
48 holdout taskと6 operator probeのevent digestを
`artifacts/research_direction_joint_synthesis_blind_validation/2026-09-25/`へ固定した。未使用H5 snapshotと
H4 opt2 compiler contextの2 stratumは48/48 holdout、6/6 operator probeまで完了し、両方で全6 gateが
通過した。結果とscopeは
[`docs/research_direction_joint_synthesis_blind_validation.md`](docs/research_direction_joint_synthesis_blind_validation.md)に記録する。
blind gate時点ではP-A v1を正式候補へ進めた。続く形式化は
`src/trotterlib/research_direction_joint_synthesis_formalization.py`、
`scripts/run_research_direction_joint_synthesis_formalization.py`、
`tests/test_research_direction_joint_synthesis_formalization.py`、
[`docs/research/pa_joint_synthesis_v1_formalization.md`](docs/research/pa_joint_synthesis_v1_formalization.md)を
一組として読む。DP最適性は有限proxy候補内で形式化できたが、全54 recordが1 run 1 segmentで、
interval分割と非零Taylor-order構造は未検証だった。P-Aを条件付き候補へ狭め、次はこの2点を明示的な
one-segment baselineに対して判別する。H12、長RPE総cost、coupling/noiseは次の必須作業ではない。

続く非退化mechanism validationは
`src/trotterlib/research_direction_joint_synthesis_mechanism_validation.py`、
`scripts/run_research_direction_joint_synthesis_mechanism_validation.py`、
`tests/test_research_direction_joint_synthesis_mechanism_validation.py`、
[事前登録](docs/research/pa_joint_synthesis_mechanism_validation_preregistration.md)、
[結果文書](docs/research_direction_joint_synthesis_mechanism_validation.md)を一組として読む。
training/blindをDF fragmentで分離したforced-support order-2全30 taskでcandidateと一区間baselineの
plan・compiled metricが完全一致し、run内分割は0件だった。P-Aのinterval claimを停止してP-Cへ戻る。
H12、長RPE総cost、coupling/noiseは次の必須作業ではない。


続くP-C tracking・breakdown validationは
`src/trotterlib/research_direction_geometry_tracking_breakdown.py`、
`scripts/run_research_direction_geometry_tracking_breakdown.py`、
`tests/test_research_direction_geometry_tracking_breakdown.py`、
[事前登録](docs/research/pc_geometry_tracking_breakdown_preregistration.md)、
[結果文書](docs/research_direction_geometry_tracking_breakdown.md)、
`artifacts/research_direction_geometry_tracking_breakdown/2026-09-25/`を一組として読む。
固定8 geometry・2 policyの16/16 taskを完了した。追跡prefixは全点で独立先頭3 fragmentと同一、
blind coefficientは3/5、pairは1/4だけが固定誤差基準内で、診断正解率は50%だった。
7 gate中3 gate通過でcurrent H4 familyのP-Cを主研究候補から停止した。
先行P-C pilotの訂正版exact-energy artifactも同時に参照する。

P-D energy係数・random-tail負担Pareto監査は
`src/trotterlib/research_direction_energy_tail_pareto.py`、
`scripts/run_research_direction_energy_tail_pareto.py`、
`tests/test_research_direction_energy_tail_pareto.py`、
[事前登録](docs/research/pd_energy_tail_pareto_preregistration.md)、
[結果文書](docs/research_direction_energy_tail_pareto.md)、
`artifacts/research_direction_energy_tail_pareto/2026-09-25/`を一組として読む。
development `L_D=3`とblind `L_D=4`で同じselection reversalを確認し、7 gate全てを通過した。
ただしexact two-block pilotであり、負時間RTEと内部`H_D`誤差の次gateを通るまでは
`P-D-conditional-candidate`として扱う。

P-D現実化Go/No-Goは`research_direction_pd_realization.py`と同名runner/test、
[事前登録](docs/research/pd_realization_go_no_go_preregistration.md)、
[結果文書](docs/research_direction_pd_realization.md)、
`artifacts/research_direction_pd_realization/2026-09-25/`を一組として読む。

続くS0/S1公平再最適化は
`src/trotterlib/research_direction_pd_fair_comparison.py`、
`scripts/run_research_direction_pd_fair_comparison.py`、
`tests/test_research_direction_pd_fair_comparison.py`、
[S0契約](docs/research/pd_primary_research_contract.md)、
[既知baseline](docs/research/pd_prior_art_and_baselines.md)、
[S1事前登録](docs/research/pd_s1_fair_comparison_preregistration.md)、
[結果文書](docs/research_direction_pd_fair_comparison.md)、
`artifacts/research_direction_pd_fair_comparison/2026-09-26/`を一組として読む。
B1b/B2/B4は同じ選択となりCase C/Dの証拠は得られず、B1aの`m_D`上限依存により
`stop_s1_undetermined_boundary_no_go_decision`で停止した。

その固定artifactの事後再解析は
`src/trotterlib/research_direction_pd_s1_posthoc.py`、
`scripts/run_research_direction_pd_s1_posthoc.py`、
`tests/test_research_direction_pd_s1_posthoc.py`、
[外部review](pd_s1_review_5c331f0.md)、
[事後計画](docs/research/pd_s1_posthoc_reanalysis_plan.md)、
[結果文書](docs/research_direction_pd_s1_posthoc.md)、
`pd_s1_posthoc_reanalysis_v1.json`を一組として読む。一次Case Bは保存し、B1b/B2/B4だけを
固定候補集合でCase A相当と事後解釈する。P-D S2は開始せず、R3も未採用である。

R3の次段判断は
[R3先行研究監査と条件付き最小研究契約](docs/research/r3_prior_art_and_minimal_contract.md)を読む。
広いsplit/error/cost最適化は既存研究との重複が強いためNo-Goである。R3-Sはselectiveな認証・棄却へ狭めて監査したが、一般certified multi-fidelity法との差分と
quantum-specific保証を固定できなかった。`STOP_R3_NO_METHOD_DELTA`でR3を停止し、数値pilotを開始しない。

finite-RTE phase/radius分離FR-1は
`src/trotterlib/finite_rte_phase_amplitude.py`、
`scripts/run_finite_rte_phase_amplitude.py`、
`tests/test_finite_rte_phase_amplitude.py`、
[FR-0契約](docs/research/finite_rte_phase_amplitude_contract.md)、
[scoped先行研究監査](docs/research/finite_rte_phase_amplitude_prior_art.md)、
[FR-1事前登録](docs/research/finite_rte_phase_amplitude_fr1_preregistration.md)、
[結果文書](docs/finite_rte_phase_amplitude_validation.md)、
`artifacts/finite_rte_phase_amplitude/2026-09-26/`を一組として読む。
33条件・99状態でG0/G1/G3/G4を通過し、G2は不通過だった。判定は
`GO_FR2_MECHANISM_ONLY`であり、利用可能情報による実用的GOではないため旧FR-2は開始しない。
その後のFR-R0で正scalar分離、情報層、強いbaseline、非一様系のdecision gateを正式契約へ固定した。
FR-R1a事後解析は完了し、正scalar処理後にFR固有の片側認証差が残らない
`POSTHOC_SCALAR_EXPLAINS_OLD_GAIN`となった。続くFR-R1b非一様4×4検証では、同情報FR境界が
norm境界より厳しい8 witnessを得た一方、固定予算の片側認証差は0だった。現行判断は
`MECHANISM_ONLY_NO_PRACTICAL_GO`で、強制停止中である。

FR-1後の再設計は、[提案文書](fr1_revised_research_plan_20260926.md)と
[FR-R0正式契約](docs/research/fr_revision_scalar_structure_contract.md)を読む。後者が現行の規範で、
正scalar分離、I0/I1/I2、共通$\gamma$と最適化比較、FR-R1の事前登録要件を固定する。
statusは`FR_R1B_COMPLETE_MECHANISM_ONLY_NO_PRACTICAL_GO_MANDATORY_STOP`である。
[FR-R1a実装](src/trotterlib/fr_revision_fr1a_posthoc.py)、
[runner](scripts/run_fr_revision_fr1a_posthoc.py)、
[test](tests/test_fr_revision_fr1a_posthoc.py)、
[FR-R1a事後計画](docs/research/fr_revision_fr1a_posthoc_plan.md)と
[FR-R1a結果](docs/fr_revision_fr1a_posthoc.md)、
[artifact](artifacts/fr_revision_fr1a_posthoc/2026-09-26/)、
[FR-R1b事前登録](docs/research/fr_revision_nonuniform_preregistration.md)、
[FR-R1b実装](src/trotterlib/fr_revision_nonuniform.py)、
[runner](scripts/run_fr_revision_nonuniform.py)、
[test](tests/test_fr_revision_nonuniform.py)、
[FR-R1b結果](docs/fr_revision_nonuniform.md)、
[artifact](artifacts/fr_revision_nonuniform/2026-09-27/)を一組として読む。
FR-R2は開始していない。

FR-R1b後の現行方針は、[研究主張・証明義務・完成原稿契約](docs/research/fr_research_claim_and_manuscript.md)
を読む。正scalar分離と方向依存補正をC1、同じ情報層でのstrict improvementと改善不能条件をC2、
resource designを条件付きC3として分離する。新規性とT1--T4の証明監査が終わるまでは、既存artifact
だけを用いて原稿を完成させ、FR-R2、H4/H12、長RPE、新しいgridへ進まない。

## ファイルの状態区分

### 現行の正本

- `docs/research/研究概要・現状.md`
- `docs/research/研究目的・研究課題.md`
- `docs/research/研究方法・解析手順.md`
- `docs/research/数値実験・評価計画.md`
- `VALIDATION_STATUS.md`
- `artifacts/validation_manifest.json`
- `docs/research/fr_revision_scalar_structure_contract.md`（FR-R0の現行比較契約）
- `docs/research/fr_research_claim_and_manuscript.md`（FR-R1b後の研究完成フェーズ契約）

### 現行実装・検証

- `docs/research/pr2_s0_s1_execution_amendment_v3.md`、
  `artifacts/pr2_s1_s3_preregistration/2026-09-28/pr2_s0_s1_authorization_manifest_v3.json`
  （S0実行、条件付きS1 correctness、S1後mandatory STOPの結果前許可）
- `src/trotterlib/pr2_s0_s1_validation.py`、`scripts/run_pr2_s0_s1_validation.py`、
  `tests/test_pr2_s0_s1_validation.py`（snapshot/identity/corrected-estimator gateとS1 correctness-only経路。
  S2/S3 commandは実装しない）
- `src/trotterlib/pr2_v4_s2_development_validation.py`、`scripts/run_pr2_v4_correctness.py`、
  `scripts/run_pr2_s2_development.py`、`tests/test_pr2_v4_s2_development_validation.py`
  （別snapshot系列のV4 correctnessとdevelopment-only S2比較のserial経路）
- `src/trotterlib/pr2_v4_s2_parallel_execution.py`、
  `scripts/run_pr2_s2_development_parallel.py`、
  `tests/test_pr2_v4_s2_parallel_execution.py`、
  `docs/pr2_s2_parallel_execution.md`（同じS2 cellと段階barrierを保つbounded CPU並列実行層。
  実H4 S2結果は`docs/pr2_v4_s2_development_validation.md`と
  `artifacts/pr2_v4_s2_development/2026-09-29/`で追跡する）
- `docs/research/pr2_matched_accuracy_prior_art_gate_v1.md`、
  `docs/research/pr2_matched_accuracy_resource_contract_v1.md`、
  `docs/research/pr2_matched_accuracy_m1_implementation_contract_v1.md`、
  `docs/research/pr2_matched_accuracy_m1_preexecution_amendment_v2.md`、
  `docs/research/pr2_matched_accuracy_m1_execution_authorization_v1.md`、
  `docs/research/pr2_matched_accuracy_m1_execution_authorization_v1_1.md`、
  `docs/research/pr2_m1_a_selection_limited_external_review_request_3c1831e.md`、
  `docs/research/pr2_matched_accuracy_m1_b1_bounded_compile_contract_v1.md`、
  `docs/research/pr2_matched_accuracy_m1_b1_execution_contract_amendment_v2.md`、
  `docs/research/pr2_matched_accuracy_m1_b1_execution_authorization_v1.md`、
  `docs/research/pr2_m1_b1_execution_authorization_external_review_request_8fc2400.md`、
  `docs/research/pr2_m1_b1_preexecution_external_review_request_1228168.md`
  （S2後の新規性gate、M1前研究契約、zero-compute実装契約。候補identity、16-cell selector、schema、
  seed規則、追加prior-art gate、M1-A/M1-B hard barrier、compile-free M1-A予算を固定済み）
- `src/trotterlib/pr2_matched_accuracy_m1_contract.py`、
  `src/trotterlib/pr2_matched_accuracy_m1_precompile_barrier.py`、
  `src/trotterlib/pr2_matched_accuracy_m1_execution.py`、
  `src/trotterlib/pr2_matched_accuracy_m1_b1_contract.py`、
  `src/trotterlib/pr2_matched_accuracy_m1_b1_execution.py`、
  `scripts/run_pr2_matched_accuracy_m1_contract.py`、
  `scripts/run_pr2_matched_accuracy_m1_precompile_barrier.py`、
  `scripts/run_pr2_matched_accuracy_m1_a.py`、
  `scripts/run_pr2_matched_accuracy_m1_b1_contract.py`、
  `scripts/run_pr2_matched_accuracy_m1_b1.py`、
  `tests/test_pr2_matched_accuracy_m1_contract.py`、
  `tests/test_pr2_matched_accuracy_m1_precompile_barrier.py`、
  `tests/test_pr2_matched_accuracy_m1_execution.py`、
  `tests/test_pr2_matched_accuracy_m1_b1_contract.py`、
  `tests/test_pr2_matched_accuracy_m1_b1_execution.py`、
  `docs/pr2_matched_accuracy_m1_a_validation.md`、
  `docs/pr2_matched_accuracy_m1_b1_result_validation.md`、
  `artifacts/pr2_matched_accuracy_m1_contract/2026-09-29/`、
  `artifacts/pr2_matched_accuracy_m1_execution/2026-09-30/`、
  `artifacts/pr2_matched_accuracy_m1_b1_contract/2026-09-30/`、
  `artifacts/pr2_matched_accuracy_m1_b1_execution/2026-09-30/`、
  `artifacts/pr2_matched_accuracy_m1_b1_result_validation/2026-10-03/`
  （候補列挙、synthetic barrier、development-only dense signal M1-A、194+16 cellのbounded compile
  plan/source/result/validation。science runnerはcompile map完成後に研究四分岐を自動判定せずreview待ちで
  停止し、別validatorがcheckpoint/cache再集計と研究reviewを行う。このstageではheld-out未承認）
- `src/trotterlib/pr2_matched_accuracy_m1_b1_result_validation.py`、
  `scripts/run_pr2_matched_accuracy_m1_b1_result_validation.py`、
  `tests/test_pr2_matched_accuracy_m1_b1_result_validation.py`
  （保存済みM1-A/M1-B1 artifact、全checkpoint、candidate別SQLite cacheをread-onlyで検査し、
  actual Pareto、旧selector、fixed-q=8、proxy、状態準備感度を再集計する。分子snapshot/held-outは読まない）
- `docs/research/pr2_matched_accuracy_m2_held_out_transfer_contract_v1.md`、
  `docs/research/pr2_matched_accuracy_m2_transfer_contract_amendment_v2.md`、
  `docs/research/pr2_m2_transfer_contract_external_review_request_06b2c32.md`、
  `src/trotterlib/pr2_matched_accuracy_m2_transfer_contract.py`、
  `scripts/run_pr2_matched_accuracy_m2_transfer_contract.py`、
  `tests/test_pr2_matched_accuracy_m2_transfer_contract.py`、
  `artifacts/pr2_matched_accuracy_m2_transfer_contract/2026-10-04/`
  （M1-B1の5構成、判定量、重大underestimate、4 status、196-wrapper上限をzero-compute固定する。
  v1/draftは履歴として保存し、正式v2はusable B2にPareto support/ratioを統一してcommit固定した。
  zero-compute contract runner自身はheld-outを開かず科学実行を認可しない。後続M2結果は下記の専用入口）
- `docs/research/pr2_s0_reproduction_stop_c644925.md`、
  `docs/research/pr2_s0_external_review_request_c644925.md`、
  `artifacts/pr2_s0_s1_validation/2026-09-28/`（development byte-level hash不一致による
  `STOP_INPUT_REPRODUCTION_MISMATCH`、S1未実行、held-out signal/cost/ranking未開封、および外部レビュー依頼）
- `src/trotterlib/` のDF、RTE、RPE、compiled-cost関連モジュール
- 対応する `scripts/run_*.py`、`tests/test_*.py`、`docs/*_validation.md`
- manifestに登録され、statusと限界が明示されたartifact

### 歴史的・補助的資料

- `abe_trotter_project.ipynb`：旧来の高次PF解析ノートブック
- `Partial Randomized Study Protocol.md`：初期計画と意思決定の履歴
- `codex-inst.md`：過去の実装依頼メモ
- `README_partial_randomized_pf.md`：旧screeningを含む実装経路の説明
- `docs/main_audit_20260801.md`：特定時点の監査記録
- `partial_randomized_trotter_validation_review_898da848.md`：commit `898da848`を対象にした外部GPTレビュー。
  現在のP-A/P-C/P-B判断より前の評価であり、一次証拠または現行仕様ではない
- `partial_randomized_trotter_research_redesign_20260925.md`：上記レビューを受けたテーマ再設計入力。
  pilotの着想と停止条件を確認する補助資料で、実施結果と現在の判断は正本文書を優先する
- `fr1_revised_research_plan_20260926.md`：FR-1後の再設計入力。採択済み部分の正本は
  `docs/research/fr_revision_scalar_structure_contract.md`を優先する
- `research_focus_and_completion_plan_16d4482_20260927.md`：FR-R1b後の完成方針を検討した入力資料。
  採択済みの主張階層、証明義務、停止条件は`docs/research/fr_research_claim_and_manuscript.md`を優先する

これらは削除していないが、現在の研究方針や最新結果を確定する根拠には使わない。

## GPTが回答・資料作成するときの確認事項

- 事前検証カタログの実施状況は、[`prevalidation_catalog_evidence_map.md`](docs/research/prevalidation_catalog_evidence_map.md)から専用文書・artifact・testまで追跡する。
- 「理論上の関係」「ローカル検証済み」「実装のみ」「未検証」「最終結論」を分ける。
- 数値には、分子、距離、basis、DF rank、$L_D$、時間幅、Taylor cutoff、検証範囲を添える。
- `C_use`を厳密上界と呼ばない。実行したdelta窓上の経験的包絡である。
- H4/H6の係数をH12へ外挿しない。
- PFの決定論的biasとQPE/RPEの統計誤差を混同しない。
- dirty worktreeの結果やローカルテストを、公開済み・CI固定済みの証拠と表現しない。
- スライド構成案や研究ノートの単独記述より、研究概要、検証文書、manifestを優先する。

## 保守規則

研究上の決定や検証結果が変わった場合は、次を同じ変更で更新する。

1. `docs/research/研究概要・現状.md`
2. 対応する研究方針文書または検証文書
3. 当日の研究ノート
4. 証拠構成が変わる場合は `artifacts/validation_manifest.json`

新しい検証を追加するときは、可能な限り同じ語幹で
`src/trotterlib/`、`scripts/`、`tests/`、`docs/`、`artifacts/`を対応させる。

## M2 usable B2契約修正 v2（2026-10-04）

外部reviewの修正要求を[amendment v2](docs/research/pr2_matched_accuracy_m2_transfer_contract_amendment_v2.md)へ反映した。
Pareto supportとprimary ratioは共にaccuracy-eligibleかつprimary重大underestimateのないB2だけを使う。
v1証拠・固定5構成・seed・196-wrapper上限を維持した。この契約修正時点では科学実行とheld-out accessは未認可だった。
moduleは`src/trotterlib/pr2_matched_accuracy_m2_transfer_contract.py`、runnerは
`scripts/run_pr2_matched_accuracy_m2_transfer_contract.py`、testは
`tests/test_pr2_matched_accuracy_m2_transfer_contract.py`、schema/planは
`artifacts/pr2_matched_accuracy_m2_transfer_contract/2026-10-04/`から辿れる。

## PR-2 M2科学実行コードの入口

[実装資料](docs/research/pr2_matched_accuracy_m2_transfer_execution_implementation.md)と
`src/trotterlib/pr2_matched_accuracy_m2_transfer_execution.py`、
`scripts/run_pr2_matched_accuracy_m2_transfer.py`、
`tests/test_pr2_matched_accuracy_m2_transfer_execution.py`、
`artifacts/pr2_matched_accuracy_m2_execution/2026-10-04/`を対応させる。
契約v2と正式planはcommit固定済み。source/authorization/環境gateはheld-out読み込みより前に置き、
旧draftは履歴だけであり、正式planはCOMMIT_BOUND。source/plan固定後に別authorizationと最終reviewを経て
一回のM2を実行した。科学sourceは変更せず、結果後のmandatory STOPを維持する。

## PR-2 M2最終実行前レビューの入口

[実行authorization](docs/research/pr2_matched_accuracy_m2_transfer_execution_authorization_v1.md)と
[最終review依頼](docs/research/pr2_m2_execution_authorization_external_review_request_90a9f24.md)、
`artifacts/pr2_matched_accuracy_m2_execution/2026-10-04/authorization_audit_v1.json`を一組として読む。
actual science sourceとplanを変更せず別authorizationをcommitし、review待ちで一旦停止した。
review承認と利用者指示を得た後に一度実行し、現在は`TRANSFER_SUPPORTED`後の研究方針review待ちである。

## PR-2 M2 held-out結果の入口

[結果照合](docs/pr2_matched_accuracy_m2_transfer_result_validation.md)と
`artifacts/pr2_matched_accuracy_m2_transfer_execution/2026-10-04/`のresult、runner manifest、complete marker、
launch/post-execution auditを対応させる。196 wrapper、5構成、source128、pre/post84＋134 testsを検査した
result commitへ収録するlocal execution evidenceであり、immutable CIではない。`.runtime`とone-shot registryはcommit対象ではない。
固定5構成のtransfer支持だけを解釈し、全status後STOP・追加科学計算未認可を維持する。

## Track B G1限定source準備（2026-10-09）

[G1 source review](docs/tracks/algorithm_codesign/g1_source_review_20261009.md)から、
[独立構造監査・controller](scripts/tracks/algorithm_codesign/g1_decision_packet/README.md)、
[入口](scripts/tracks/algorithm_codesign/run_g1_decision_packet.py)、
[focused tests](tests/tracks/algorithm_codesign/test_g1_source_preparation.py)、
[準備artifact](artifacts/track_b_g1_source_preparation/2026-10-09/evidence_manifest_v1.json)へ辿る。
59件のlocal off-domain testsが通過。本構造監査・固定8人工LPは未実行で、別のsource-bound明示指示を待つ。
既存source/contract/result/marker/STOPと共有APIを保持する。全outcomeでSTOPしGPT G1へ戻す。

## Track B G2保存値診断（2026-10-09）

[GPT G2 handoff](docs/tracks/algorithm_codesign/g2_saved_diagnostic_handoff_20261009.md)と
[独立数学監査](docs/tracks/algorithm_codesign/g2_independent_math_audit_20261009.md)、
`scripts/tracks/algorithm_codesign/g2_saved_diagnostic.py`、
`tests/tracks/algorithm_codesign/test_g2_saved_diagnostic.py`、
`artifacts/track_b_g2_saved_diagnostic/2026-10-09/evidence_manifest_v1.json`を対応させる。
固定保存表504 profileのpost-hoc/development診断。旧G1/RA-D0結果・markerは保持し、
新registered LP/science/synthesisは0。完了後STOPし研究価値・次scopeをGPT G2へ返す。


## Track B G3有限law・known return比較（2026-10-09）

[GPT G3 handoff](docs/tracks/algorithm_codesign/g3_finite_law_handoff_20261009.md)、
[GPT G2 review](docs/research/track_b_G2_scientific_review_20261009.md)、
`scripts/tracks/algorithm_codesign/g3_finite_law.py`、`g3_return_comparator.py`、`audit_g3_saved_outputs.py`、
`tests/tracks/algorithm_codesign/test_g3_finite_law.py`、`test_g3_return_comparator.py`、
`artifacts/track_b_g3_finite_law/2026-10-09/evidence_manifest_v1.json`を対応させる。
承認されたdevelopment保存値288 profile/6,912 lawsと、条件付きreturn12合成/12 native event/162 lawsを完了。
限定poolでT/1Q差を記録、全B2任意混合の最適性・旧RA-D0 witnessではない。mandatory STOP、次判断はGPT G3。


## Track B G4条件付き分離・matched CTS（2026-10-09）

[G4結果/GPT handoff](docs/tracks/algorithm_codesign/g4_results_and_gpt_handoff_20261009.md)、
[独立証明](docs/tracks/algorithm_codesign/g4_A_independent_proof_20261009.md)、
[CTS結果前仕様](docs/tracks/algorithm_codesign/g4_B_matched_CTS_contract_20261009.md)、
[G4-C設計のみ](docs/tracks/algorithm_codesign/g4_C_next_validation_design_20261009.md)、
`scripts/tracks/algorithm_codesign/g4_independent_certificate.py`、`g4_cts_specialization.py`、
`g4_matched_cts.py`、`audit_g4_saved_outputs.py`、2 focused test files、
`artifacts/track_b_g4_conditional_separation/2026-10-09/evidence_manifest_v1.json`を対応させる。
固定class分離とI1 CTS対照を分け、全工程STOP・研究採否はGPT。共有API変更なし。


## 2026-10-10 Track B G5：固定辞書の終了認証（最新追記）

GPT G4 reviewを利用者が採用し、現固定toy・同辞書の実用優位実験主線を区切る。
[R0/G1/G4/G5 claim map](docs/tracks/algorithm_codesign/g5_claim_evidence_map_20261010.md)、
[G5 handoff](docs/tracks/algorithm_codesign/g5_results_and_gpt_handoff_20261010.md)、
[結果前scope](docs/tracks/algorithm_codesign/g5_fixed_dictionary_closure_scope_20261010.md)、
[static access inventory](docs/tracks/algorithm_codesign/g5_static_access_inventory_20261010.md)。
[stdlib saved-only verifier](scripts/tracks/algorithm_codesign/g5_fixed_dictionary_closure.py)、
[off-domain tests](tests/tracks/algorithm_codesign/test_g5_fixed_dictionary_closure.py)、
[manifest](artifacts/track_b_g5_fixed_dictionary_closure/2026-10-10/evidence_manifest_v1.json)、
[研究note](docs/research/研究ノート/2026-10-10_track_b_g5_fixed_dictionary_closure.md)を入口とする。
x1/4・7 prototypes/3 precision・6頂点252 profiles、同Bernstein/L1-charge digital classへの保守的三下界を同CTS lawが下回る。
新LP/synthesis/circuit/matrix/DF/分子/NPZ/GPUは0。限定理論成果は保持、Track B全体の終了・次method採択ではない。
mandatory STOP。上記より前のpending/最新記述は当時の履歴としてそのまま保持する。


## 2026-10-10 Track B G6：return generatorの独立数学監査（最新追記）

利用者採用の[GPT G5研究方針](docs/research/track_b_G5_research_direction_review_20261010.md) §19に基づく新しい探索的数理監査。
[G6結果/GPT引継ぎ](docs/tracks/algorithm_codesign/g6_results_and_gpt_handoff_20261010.md)、
[一般証明](docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md)、
[一次文献・同値性](docs/tracks/algorithm_codesign/g6_prior_art_and_method_delta_20261010.md)、
[finite-bit/access](docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md)。
[局所prototype](src/trottertracks/algorithm_codesign/return_aggregation.py)、
[technical checker](scripts/tracks/algorithm_codesign/g6_return_generator_audit.py)、
[16 focused tests](tests/tracks/algorithm_codesign/test_g6_return_aggregation.py)、
[記録](artifacts/track_b_g6_return_generator_audit/2026-10-10/)。
ideal数式を確認、母関数/zero-fillは既知components。native費用・独立新規性は未確定。科学実行0、mandatory STOP。


## Track B G7（2026-10-10、source preparation）

GPT G6 review §12の限定予算/control/implementation economics。
[数学・実行契約](docs/tracks/algorithm_codesign/g7_mathematical_and_execution_contract_20261010.md)、
[source](src/trottertracks/algorithm_codesign/g7_generator.py)、
[runner](scripts/tracks/algorithm_codesign/g7_budget_control_economics.py)、
[tests](tests/tracks/algorithm_codesign/test_g7_budget_and_control.py)、
[contract](artifacts/track_b_g7_budget_control_preparation/2026-10-10/contract_v1.json)。
2 known development inputs/4 arms/24 fixed keys、conditional controlled-Q/Rz/CPU vector、18 focused tests。
この追記はsource preparation。結果は別manifestへ追加し、完了/失敗後mandatory STOP。


## 2026-10-10 Track B G7：取得完了・mandatory STOP（最新追記）

[G7結果/GPT引継ぎ](docs/tracks/algorithm_codesign/g7_results_and_gpt_handoff_20261010.md)。
`G7_LIMITED_IMPLEMENTATION_ECONOMICS_COMPLETE_AWAITING_GPT_REVIEW`。source ab2549f41b3546fb3940342a2162dd9ee93699c4、24 keys/8 rows、strict error PASS、retry0。
固定P5では登録3対照後にもconditional期待T減少、P3ではclosed-form対照が小さい。
hard shot cap・classical generation/angle acquisition・未指定provider costを併記。
新規性/主method/次stage未採択、G5閉鎖/G6原証拠とmarker保持、GPT判断へ戻す。


## Track B G8（2026-10-10、source preparation）

[G8 proof/contract](docs/tracks/algorithm_codesign/g8_proof_contract_and_on_demand_scope_20261010.md)：採用GPT G7 review §11に基づく限定確認。
2 known development inputs/4 same production laws、finite-provider parameter、分離failure配分、
on-demand strict Rzと対称bounded cache。G7のstatus/point comparison/consumed markerを保持。
17 off-domain focused tests、source preparation時点のnew native acquisition0。一束後mandatory STOP。
source：`src/trottertracks/algorithm_codesign/g8_bounds.py`、`g8_pipeline.py`、`g8_reference_audit.py`。
runner：`scripts/tracks/algorithm_codesign/g8_on_demand_provider_budget.py`、tests：`tests/tracks/algorithm_codesign/test_g8_on_demand_and_bounds.py`。
artifacts：`artifacts/track_b_g8_on_demand_preparation/2026-10-10/` と新result directory。


## Track B G8 completed evidence (2026-10-10)

- [Results and GPT handoff](docs/tracks/algorithm_codesign/g8_results_and_gpt_handoff_20261010.md): on-demand20/8rows, hypothetical provider budget, saved-only14checks.
- [G8 result inventory](artifacts/track_b_g8_on_demand_result/2026-10-10/v1/evidence_manifest_v1.json); mandatory STOP, no native-method adoption.


## Track B G9 source preparation（2026-10-10）

[Fixed proof/scope](docs/tracks/algorithm_codesign/g9_p5_matched_native_contract_20261010.md)：GPT G8 review §14を採用。known P5/指定3-qubit provider、6 direct+5 helper診断、19 keys/新CTS1 key、23 focused tests。source固定後一束のみ、終了後mandatory STOP。

## G9 one-shot STOP（2026-10-10）

`G9_TECHNICAL_INCONCLUSIVE`：epsilon引数のFraction→mp.mpf変換で停止、native比較0 row。
新規helper attempts1 / 新規sequence取得0 / retry0。source・contract・過去941pathは不変。
23 focused testsと23 saved-output checksは準備/整合証拠で、native資源の科学結果ではない。
[G9 failure・GPT handoff](docs/tracks/algorithm_codesign/g9_results_and_gpt_handoff_20261010.md)。mandatory STOP、次の研究判断はGPTへ返す。

## G9 v2 API-boundary source preparation（2026-10-10）

[G9 v2準備/入口](docs/tracks/algorithm_codesign/g9_v2_api_boundary_source_review_20261010.md)。精度値の型接続を最小修正、19 stub-only/launch tests PASS。
46科学条件・同19-key inventory・旧v1 source/result/markerは保持。
新実合成・登録matrix/予算/科学実行0、v2 marker absent、別authorization pending。
独立branchで資料公開後STOPし、新source-bound明示認可を待つ。

## G9 v2 one-shot completed / STOP（2026-10-10）

[G9 v2 results/GPT handoff](docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md)。`G9_MATCHED_NATIVE_RESOURCE_MAP_COMPLETE`、11 rows/22 axes、19 keys（new1/reuse18）、retry0。
known P5/指定3-qubit providerの登録direct6方式でclosed P5のT intercept/Kが小さく、CTSのCXは小さい。
1,866 event accounting、saved-only25 checks PASS。旧v1結果/marker、critical80/protected982は不変。
実量子shots/trajectory/DF/分子/NPZ/GPU/LPは0、source-bound local evidence。次の科学実行・採択はGPT判断、mandatory STOP。


## Track B G9 v2科学review受領（2026-10-10）

[採用review](docs/research/track_b_G9_v2_scientific_review_20261010.md) / [G10 bundle範囲](docs/tracks/algorithm_codesign/g9_v2_review_intake_and_g10_scope_20261010.md)。
return集約family限定継続、closed P5を低次数の標準実装へ。
同p/x/providerのm=3/5/7比較が次のscope、m5は保存anchor。
今回はdocs-only受領、CTS下界独立検証/G10実装・契約・実行は未実施。旧G9 result/marker/STOP不変。
[受領manifest](artifacts/track_b_g9_v2_review_intake/2026-10-10/intake_manifest_v1.json)。G10一束後mandatory STOP、科学判断はGPT。
