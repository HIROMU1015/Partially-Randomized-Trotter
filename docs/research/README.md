# 研究計画資料

## 2026-10-06 H4 signal/compile INPUT_BOUND草案・容量不足STOP

run02の6凍結入力へplanを結合し、source19/218 templates/compiler/run ID/outputを不変に保った。
stage必要3.5 GiB/560000 inodesに対しavailable3.419376 GiB、約82.6 MiB不足。CPU候補6 core/memory/quota観測はPASS。
全72h監視・74784 records・149569 ledger deltas・全8KiB worker logs・1308 signal files・journal/temp/directory余裕を含む。
CPU [3,5,6,7,8,9]・6 worker・own-run mask0x3e8は次段の提案、review=false、利用者の別stage承認/明示launch未取得。
signal/seed/sampling/build/compile/transpile/taskset/worker/GPU/共有環境・他job変更0、入力再生成0。
[次段scope・容量](track_a_h4_signal_compile_plan_review.md)と[資料入口・承認対象](../../artifacts/resource_applicability/track_a_h4_signal_compile_plan_review/2026-10-06/README.md)を参照する。
`H4_SIGNAL_COMPILE_PLAN_PREPARED_STORAGE_BLOCKED_STOP`。容量と別認可成立後もfresh検査不合格なら起動せず、map後MAP_COMPLETE_STOP。
scientific runtimeはcommitしない。旧10GiB案/旧source/旧bundle/入力生成完了と以下の履歴は保持する。

## 2026-10-06 H4入力生成run02・6入力freeze完了STOP

利用者の明示再実行指示でworker bootstrapのstdlib signal shadowを-Pで修正し、run01を保存してrun02を別固定した。
CPU [3,5,6,7,8,9]・6 workers・own-run限定、fresh resource/容量/quota検査PASS。
H4 linear/STO-3G/DF12の追加6距離入力を一度生成しfreeze完了。NPZ6 bytes SHAを照合し、own driver/worker残存0。
sourceは049e69919af16ad29a67a217dc7a407d6b1754a6。科学/seed/compiler/resource条件は不変、source19変更はgates/worker起動だけ。
`INPUTS_FROZEN_STOP`、next_stage_authorized=false、mandatory_stop=true。signal/compile/GPU/追加transpile0、共有環境・他job変更0。
[修正・完了scope](track_a_h4_worker_bootstrap_run02.md)と[完了報告](../../artifacts/resource_applicability/track_a_h4_worker_bootstrap_run02/2026-10-06/COMPLETION_REPORT_v1.md)を参照する。scientific runtimeはcommitしない。以下は各時点の履歴。

## 2026-10-06 H4入力生成stage容量確認・最終承認待ち

容量準備の判定は「足りる」。6入力生成→freeze→STOPの必要量3GiB/260000 inodesに対し、
2026-10-06 16:58:53 JSTのnonroot available約3.615GiB、225817022 inodes、user/group/project quota非有効をread-only確認。
32保存配列/NPY・ZIP overhead/64MiB IPC上限/temp-final/72h監視259202 files/journal/metadata余裕を含む。
全campaign10GiBはcharge capとして維持し、旧全量空き確保案を履歴に保存した上でstage-specific補足を追加した。
source19/plan/auth/approved=false reviewはbyte-identical、CPU [3,5,6,7,8,9]・6 worker・own-run mask0x3e8は未承認。
CPU使用許可/独立最終review/明示launch/fresh CPU・memory・pressure/OOM・容量検査が残る。signal/compile容量は別認可。
[容量根拠](track_a_h4_input_generation_stage_storage_review.md)と[最終承認資料入口](../../artifacts/resource_applicability/track_a_h4_input_generation_storage_review/2026-10-06/README.md)を参照する。
`H4_INPUT_GENERATION_STAGE_STORAGE_CONFIRMED_AWAITING_FINAL_APPROVAL`で公開後STOP。
新科学/追加transpile/taskset/worker/GPU/共有環境・他job変更0。旧10GiB案を含む以下は当時の履歴。

## 2026-10-06 H4入力生成 CPU/launch最終案・利用者承認未取得

提案CPU[3,5,6,7,8,9]、6 worker、異なる6 physical core・NUMA0。約3秒の受動負荷sampleで各core busy0%。
source19 pathsとplan v2 bytes/source_rootは不変。authは候補CPU集合だけ、reviewはauth digestだけ変更しapproved=falseを保持。
own新規runだけにtaskset maskを指定する未実行commandを用意した。CPU許可/専有予約/独立review/明示launchは未取得。
memory/context read-only確認は成功。filesystem空き約3.717GiBは総上限10GiB全量確保案に未達で、launch容量条件は未解決。
12 metadata gate tests PASS、fail/error/skip0。観測01/02の失敗logと03の容量未解決記録を保持し、追加transpile0/旧28件不変。
[提案資料](track_a_h4_input_generation_cpu_launch_proposal.md)と[bundle・一括承認判断](../../artifacts/resource_applicability/track_a_h4_geometry_input_generation_cpu_launch_proposal/2026-10-06/README.md)を参照する。
`H4_INPUT_GENERATION_CPU_LAUNCH_PROPOSAL_FROZEN_AWAITING_APPROVAL`で公開後STOP。taskset/科学/worker/GPU/共有環境・他job変更0。

## 2026-10-06 H4入力生成 resource observer修正・未承認草案再固定

真のv2 hierarchy rootをnamespace/mount/所属から判定し、rootの非root memory interface要求を修正した。
全可視非root祖先の制限・pressure/OOMは保持し、非root欠測/不明namespace/hidden mountはSTOP、host-only fallbackなし。
observer33 zero-science tests PASS、実read-only観測成功。準備観測available約981.826GiB、PSI/OOM0はlaunch成立ではない。
production変更はresources observerとgateのnew audit pathだけ。科学/並列/seed/compiler source15件は不変。
[実装資料](track_a_h4_geometry_resource_observer_fix.md)と[source bundle](../../artifacts/resource_applicability/track_a_h4_geometry_resource_observer_fix/2026-10-06/README.md)、
[new認可草案v2](../../artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06-v2/README.md)を参照する。source固定→別草案commit、binding検査結果はv2へ記録。
requested workers6、allowed_cpus=[]、approved=false。CPU/launch context・独立最終review・明示launch未解決、実行準備完了とはしない。
`H4_INPUT_GENERATION_RESOURCE_FIX_FROZEN_AWAITING_REVIEW`で公開後STOP。科学/追加transpile/GPU/本番起動/共有環境・他job変更0。
旧bundle/科学証拠/原稿/Track Bと旧監査履歴を保存し、系列transpile28/64を維持する。

## 2026-10-06 H4 geometry 入力生成専用認可草案・実行未承認

凍結science source6a121725（17 Python＋親2件）を変えず、入力生成source-bound planとresult-prior認可草案を追加した。
requested workers6、inputs/freeze digestはnull、218 templatesを機械転記。reviewはapproved=false、allowed_cpus=[]。
CPU許可は利用者指示で未確定のまま。現在process CPU0–255を許可とみなさず、launch contextとmemory観測は未解決。
既存observerはroot cgroup memory.max欠落で停止し、実行準備完了とはしない。source/共有設定を緩和しない。
新57 zero-science gate tests PASS、fail/error/skip0。合格経路はメモリ内模擬承認だけ、追加transpile0・旧累積28/64不変。
[実装・停止条件](track_a_h4_geometry_input_generation_authorization_draft.md)と[bundle・最終レビュー入口](../../artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06/README.md)を参照する。
`H4_INPUT_GENERATION_AUTHORIZATION_DRAFT_FROZEN_AWAITING_REVIEW`で公開後STOP。科学/GPU/本番起動/共有環境・他job変更0。
有効execution authorization0、final review/利用者の明示launch未実施。入力生成・本計算・signal/compile認可・H6/Track Bへ進まない。

## 2026-10-06 H4 geometry compile並列source再固定・科学未実行

compileの逐次waitをadmitted worker数以下のbounded投入・回収へ変更した。処理中ownerを追跡し、
COMPLETEとidentity/digest検査後だけ再利用する。trajectory/axis順・weight、科学scope・seed/compilerは不変。
[実装資料](track_a_h4_geometry_parallel_source_implementation.md)と[new bundle](../../artifacts/resource_applicability/track_a_h4_geometry_parallel_source/2026-10-06/README.md)を現在の入口とする。
既存94＋並列回帰17＝111 synthetic tests PASS、fail/error/skip0。今回transpile3、旧25＋新3＝28/64。
fake futures/mock workersだけで制御を検査し、実worker/production性能は未検証。旧bundle/audit/契約・保存証拠は不変。
SOURCEとsourceを変更しないREVIEWの2 commitを分ける。分子アクセス/科学処理/GPU/本番起動/認可発行/共有環境・他job変更0。
`H4_GEOMETRY_PARALLEL_SOURCE_FROZEN_AWAITING_REVIEW`で公開後STOP。入力生成plan/auth作成・本計算・H6/Track Bへ進まない。

## 2026-10-06 H4 geometry server-native source固定・科学未実行

利用者の新指示で契約v2 D1〜D4を実装条件へ採用し、旧未承認履歴を保存した。
新namespace `src/trottertracks/resource_applicability/h4_geometry/`、二つのfuture runner、専用synthetic testsの入口は
[実装資料](track_a_h4_geometry_server_native_source_implementation.md)と
[bundle](../../artifacts/resource_applicability/track_a_h4_geometry_source/2026-10-06/README.md)。
最終94 tests pass、fail/error/skip0。失敗・再検査込みsynthetic transpile25/64、旧benchmark128再実行0。
SOURCE_COMMITとsourceを変えないREVIEW_BUNDLE_COMMITを分離し、actual blob/hashは別監査で固定する。
旧247 source・v1/v2 bundle・準備25 files・保存6 JSONは不変。分子入力/科学処理/本番runner launch/GPU/環境・他job変更/認可発行0。
`H4_GEOMETRY_SOURCE_FROZEN_AWAITING_REVIEW`で公開後STOP。別入力生成authorizationの作成へ進めるかをレビューし、今回は発行・実行しない。
以下は各milestone当時の履歴。


## 2026-10-06 H4 geometry契約v2・レビュー待ちSTOP

現在の入口は[契約v2 bundle](../../artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06-v2/README.md)。v1 commit `7c1a3d43f61c5501a9e79206b7c60933f94b1077`を保存し、
D1〜D4を具体的な採用案、memory admissionを8+8w+16 GiB、認可を入力生成→freeze STOP→別signal/compile認可へ分離した。
H4 linear/STO-3G/DF rank12、6距離・218 template・32 paired trajectories・74,784上限は不変。8 system＋ancilla1(index8)、合計9 qubits。
[pure JSON validator](../../artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06-v2/contract_validator_v2.py)と[専用合成検査](../../artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06-v2/run_contract_tests_v2.py)はreview用で、science source/runnerではない。
新規320件pass（fail/skip0）、旧129件は保存・runner再実行0。旧v1 manifestはbase blobで照合し書き換えない。
D1〜D4レビュー承認は未解決、science/source port/input generation/next stage認可false、plan未seal、mandatory STOP。
今回の公開指示は軽量契約bundleと関連文書だけのcommit/non-force push。以下は各段階当時の履歴。

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


## H4 geometry 契約準備 v1（2026-10-06・local未commit）

[準備bundle](../../artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06/README.md)は契約schema・zero-compute plan・pure JSON validatorと129合成検査の入口。
6距離、218 template/点、74,784 wrapper、最大12 workersを固定し、生成/seed/memory/wall/outputはreview待ち。
science source/runnerの追加ではなく、旧公開draft・source・科学結果は不変。本計算・port・commit/pushは未認可、STOP。


## H4候補間compile投入・12 worker明示再実行

利用者の増員再実行指示により[run03 source・認可・検査記録](track_a_h4_cross_candidate_run03.md)を追加した。候補内2回路の完了待ちで4 workerがidleとなる問題を、候補間bounded queueで修正。旧run02はworker failure STOPで全証跡を保持し、旧6入力を再生成せず利用する。49人工job/metadata testsはlocal PASSで科学的結果ではない。12 workerのfresh CPU/memory/容量/hash検査後だけ一度起動しMAP_COMPLETE_STOP。旧累積bytes/wall/actual invocationsを引継ぎ、科学条件・compiler・上限は不変。


## H4 run04：identity hash分割・5秒監視維持

[run04固定sourceと再実行binding](track_a_h4_streaming_monitor_run04.md)を追加した。run03はmonitor STOPで旧全証跡を保持。人工JSONで旧一括encoderのmonitor interval/freshnessを再現し、新streaming encoderは同じdigestで全監視PASS。65 local tests PASSは実装証拠で科学結果ではない。旧6入力を再生成せず、bytes/wall/7 consumed invocationsをcarryし、CPU12件・12 workerのfresh検査後に一度実行してMAP_COMPLETE_STOP。


## H4 run05：identity hash分割・5秒監視維持

[run05固定sourceと再実行binding](track_a_h4_lazy_identity_run05.md)を追加した。run04はmonitor STOPで旧全証跡を保持。人工JSONで旧一括encoderのmonitor interval/freshnessを再現し、新streaming encoderは同じdigestで全監視PASS。75 local tests PASSは実装証拠で科学結果ではない。旧6入力を再生成せず、bytes/wall/12 consumed invocationsをcarryし、CPU12件・12 workerのfresh検査後に一度実行してMAP_COMPLETE_STOP。


## H4 新サーバー引継ぎ（2026-10-07）

[引継ぎ資料入口](track_a_h4_new_server_handoff_20261007.md)へrun05監視STOP・累積予算・別転送対象・新hostの認可範囲を補足した。旧資料/固定sourceを保持し、本計算は新hostのCPU許可・最終review・明示launchまで開始しない。


## 2026-10-07 H4新host取得・監査

[H4新host取得・監査](track_a_h4_new_server_preparation_20261007.md)。指定commit/source19照合済み、環境18 version/45 RECORD差の解決案でSTOP。source修正・新tests・production未実施、入力/停止証拠別送待ち、allowed_cpus=[]/approved=false。旧科学結果・原稿・Track Bは保持。


## 2026-10-07 H4新host A案 source/人工検証固定・本計算未認可

[監視修正・32人工tests](track_a_h4_new_server_monitor_fix_a_20261007.md)。既存private venv不変で準備用A案採用。
旧source19は3変更/16不変、新module込み25 closure。256×256人工matrix＋9-qubitの旧/new byte/digest一致、独立observerのGIL/GC観測・I/O delay/EOF/所有・資源境界を検証。
SOURCE `b2a5ad89e8b39d72716f7ddb17d263bd0cdedb45`、production/追加transpile0、旧28/64・benchmark128保持。環境18 version差、旧45 raw-reference RECORD差保持・normalized22差を明記。入力6/freeze/runtime/control未受領。
observer AS256MiB/RSS64MiB/admission120.25GiB、容量5.5625GiB案は未承認。allowed_cpus=[]/approved=false/runtime_authorization=false/launch=null、STOP。科学成果/原稿/Track Bと旧資料を保持。


## 2026-10-08 H4本計算前準備・最終承認待ちSTOP

[統合入口・最終承認案](track_a_h4_prelaunch_preparation_20261008.md)。新host schema/profile/observer/CPU/one-shot/累積budget bindingを整備。
SOURCE `ad57d1639133f7158cce58d767b8e0aa179bf044`、32 closure、57限定人工tests PASS。core quota read-only確認、worker12＋driver/observer各1別coreを提案。
carry20/165214360 bytes/5466.188392877579秒、残74764を保持。全74784 logicalを保証する最小actual cap+20→74804案は未承認。
候補環境はprivate venv不変、18 version/旧参照対raw45・normalized22差。追加output5GiB/301000 inodes、charge約8.29GiB、copy前は暫定6GiB。
入力6/freeze/native stop proof未受領、allowed_cpus=[]/approved=false/runtime_authorization=false/未seal、science/追加transpile/GPU/共有環境・他job変更0。明示承認・final review・launch前にSTOP。


## 2026-10-08 H4受領監査・独立最終review TECHNICAL FAIL

[統合入口](track_a_h4_receipt_final_review_20261008.md)。origin79827016/SOURCEad57d163照合、source32不変、read-only live profile/資源/quotaを確認。
凍結NPZ6/freeze/native stop/controlは未受領、producer bytes/SHA manifestを含む最小転送手順を具体化。科学array読込/再生成0。
独立reviewはpidfd送信時ESRCH競合で後続cleanupが中断するP1を純mock再現しTECHNICAL FAIL。57人工PASSはこの競合を覆わない。
未受領NOT_EVALUABLE、CPU/env/observer/74804案のUNAPPROVEDと技術FAILを区別。sourceは修正せず再sealなし、flags false、carry20/165214360/5466.188392877579を保持し本計算STOP。

## 2026-10-09 H4 cleanup ESRCH修正・独立再review PASS

[新SOURCE・独立再review・残る承認条件](track_a_h4_cleanup_esrch_fix_20261009.md)。旧992c09d6から独立worktreeでP1を修正し、SOURCE `6bd1ba01cd71ec3e2071082963c9f07478dada9a`・closure33を固定。
両pidfd送信経路はESRCHだけ既退出扱いで後続cleanupを継続し、他の送信error・所有検証を保持する。
限定人工39件PASS、別担当13純mock件PASSとbinding照合でP1_SCOPE_TECHNICAL_PASS。wait/reap/FD/pipe/first STOP理由保持まで確認。
新test `tests/tracks/resource_applicability/test_h4_cleanup_esrch.py` と既存人工runnerで追跡し、source/profile/plan/auth/reviewを新SHAへ再結合した。
入力0/6・freeze/native停止証拠未受領はNOT_EVALUABLE、未seal。環境/CPU/observer/74804案は未承認、carry20/165214360 bytes/5466.188392877579秒・現actual cap74784不変。
approved=false、runtime_authorization=false、allowed_cpus=[]。追加transpile/科学actual/本番起動0。旧資料を保存して本計算STOP。

## 2026-10-09 H4全byte受領・carry合格・native終端proof待ち

[受領・binding・最終承認案](track_a_h4_byte_receipt_binding_20261009.md)。packet86691840B/全2150files/既知9SHAをbyte-only照合しPASS、NPZ6/freeze受領完了。
SOURCE `6bd1ba01cd71ec3e2071082963c9f07478dada9a`・closure33不変。ledger chain/cumulative journalからcarry20/165214360 bytes/5466.188392877579秒を保持、現cap74784・残74764。
run05 log/exact旧sourceからworker cleanup到達は推認可能だが、driver/12 workers停止後identity/残存0 native proofが不足。古いrun02停止監査をrun05proofへ流用しない。
input/profile/source/output/carryとv6 plan/auth/reviewを結合し、control82件のbasename・単一link・sender manifest mappingを照合。source条件は緩めていない。
sealed=false/approved=false/runtime_authorization=false/allowed_cpus=[]。環境/CPU/observer/累積74804案/一度のmap launchは未承認。科学array読込/新科学actual/追加transpile/共有設定変更0でSTOP。

## 2026-10-09 H4追加native停止proof合格・technical再seal

[再seal・独立最終整合review・一括承認案](track_a_h4_native_proof_seal_20261009.md)。追加JSON17521B/SHA一致、旧host/run05の13identity・2回残存0・元3証拠hashを照合して現在のnative停止条件PASS。
過去のexit code/正確な終了・reap時刻/原boot IDは未記録のままnull。今回の観測で補完せず、連続監視や歴史cleanup順の証明とも扱わない。
SOURCE `6bd1ba01cd71ec3e2071082963c9f07478dada9a`・closure33不変、profile/input/carry/control83件の固定validator合格でplan再seal、sealed=true。
approved=false/runtime_authorization=false/allowed_cpus=[]。carry20/165214360 bytes/5466.188392877579秒、現cap74784・残74764を保持。
候補environment/compiler・CPU・observer・累積actual74804案・一度のmap launchは未承認。承認による最終artifact/digest再結合とfresh resource/CPU/fs/inode/quota gateをlaunch前に確認。
科学array読込/新science actual/追加transpile/source変更/共有設定変更/本計算0でSTOP。

## 2026-10-09 H4一度のmap実行を利用者承認・最終artifact固定

[実行認可・直前gate・起動報告入口](track_a_h4_authorized_launch_20261009.md)。利用者の明示認可で候補environment/compiler採用、worker12 CPUs2/4–6/8–15・driver16・observer18、observerAS256MiB/RSS64MiB/admission120.25GiBを認可。
carry20/165214360 bytes/5466.188392877579秒を保持し、累積actualだけ74804へ+20改定。SOURCE6bd1ba01・science/compiler/options/他caps不変。
sealed/approved/runtime_authorization=true、allowed_cpusはexact14role集合。独立v8 review PASS。artifact commit後fresh CPU/memory/PSI/OOM/FS/inode/quotaとSOURCE/profile/input/carry/unusedrootを確認しPASSなら追加承認なし一度起動。
既存proof/回帰/benchmark再実行0、追加準備campaign/transpile0。oldpartial/cache/GPU/共有環境・venv・他job変更なし、完了またはfail-closed STOP後終了・retry/次stageなし。
これは認可artifact固定時点のsnapshot。実起動/PID/状態は入口への追記・外部runtime receiptで別記録する。

## 2026-10-09 H4 library cache保存先修正・再実行予算不合格

[修正・再実行条件](track_a_h4_library_cache_fix_20261009.md)。前回のOpenFermion→Matplotlib mkdir拒否を、homeの新private library cacheへprocess限定MPLCONFIGDIRを結合して修正。
driver/workerとも既存directoryのEEXIST probe以外のcache writeを拒否、29816BのSHA固定。47限定回帰＋3 import case PASS、科学array/transpile/実worker/affinity/GPU0。
SOURCE `b8b3ce6e8c98f1ec0419a7af79c5d7c5f3a3b9bb`、36 closure、science/compiler/options不変。前回費用を返却せずcarry20/4428938712B/5472.345380863175sへ結合。
累積worst charge13165893832B=12.261694GiB>承認10GiBで未seal/approved=false/runtime_authorization=false、再起動0。13GiBは未承認proposalのみ。
既承認environment/CPU/observer/actual74804を保持。cap改定・新SOURCE/gate binding/review・fresh gate後の一度再実行が残る。
private homeは共有systemと区別し、旧run/one-shot/失敗証拠・全予約課金を保持する。
