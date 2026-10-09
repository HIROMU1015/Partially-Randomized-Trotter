# プロジェクト案内

## 2026-10-06 H4 signal/compile INPUT_BOUND草案・容量不足STOP

run02の6凍結入力へplanを結合し、source19/218 templates/compiler/run ID/outputを不変に保った。
stage必要3.5 GiB/560000 inodesに対しavailable3.419376 GiB、約82.6 MiB不足。CPU候補6 core/memory/quota観測はPASS。
全72h監視・74784 records・149569 ledger deltas・全8KiB worker logs・1308 signal files・journal/temp/directory余裕を含む。
CPU [3,5,6,7,8,9]・6 worker・own-run mask0x3e8は次段の提案、review=false、利用者の別stage承認/明示launch未取得。
signal/seed/sampling/build/compile/transpile/taskset/worker/GPU/共有環境・他job変更0、入力再生成0。
[次段scope・容量](docs/research/track_a_h4_signal_compile_plan_review.md)と[資料入口・承認対象](artifacts/resource_applicability/track_a_h4_signal_compile_plan_review/2026-10-06/README.md)を参照する。
`H4_SIGNAL_COMPILE_PLAN_PREPARED_STORAGE_BLOCKED_STOP`。容量と別認可成立後もfresh検査不合格なら起動せず、map後MAP_COMPLETE_STOP。
scientific runtimeはcommitしない。旧10GiB案/旧source/旧bundle/入力生成完了と以下の履歴は保持する。

## 2026-10-06 H4入力生成run02・6入力freeze完了STOP

利用者の明示再実行指示でworker bootstrapのstdlib signal shadowを-Pで修正し、run01を保存してrun02を別固定した。
CPU [3,5,6,7,8,9]・6 workers・own-run限定、fresh resource/容量/quota検査PASS。
H4 linear/STO-3G/DF12の追加6距離入力を一度生成しfreeze完了。NPZ6 bytes SHAを照合し、own driver/worker残存0。
sourceは049e69919af16ad29a67a217dc7a407d6b1754a6。科学/seed/compiler/resource条件は不変、source19変更はgates/worker起動だけ。
`INPUTS_FROZEN_STOP`、next_stage_authorized=false、mandatory_stop=true。signal/compile/GPU/追加transpile0、共有環境・他job変更0。
[修正・完了scope](docs/research/track_a_h4_worker_bootstrap_run02.md)と[完了報告](artifacts/resource_applicability/track_a_h4_worker_bootstrap_run02/2026-10-06/COMPLETION_REPORT_v1.md)を参照する。scientific runtimeはcommitしない。以下は各時点の履歴。

## 2026-10-06 H4入力生成stage容量確認・最終承認待ち

容量準備の判定は「足りる」。6入力生成→freeze→STOPの必要量3GiB/260000 inodesに対し、
2026-10-06 16:58:53 JSTのnonroot available約3.615GiB、225817022 inodes、user/group/project quota非有効をread-only確認。
32保存配列/NPY・ZIP overhead/64MiB IPC上限/temp-final/72h監視259202 files/journal/metadata余裕を含む。
全campaign10GiBはcharge capとして維持し、旧全量空き確保案を履歴に保存した上でstage-specific補足を追加した。
source19/plan/auth/approved=false reviewはbyte-identical、CPU [3,5,6,7,8,9]・6 worker・own-run mask0x3e8は未承認。
CPU使用許可/独立最終review/明示launch/fresh CPU・memory・pressure/OOM・容量検査が残る。signal/compile容量は別認可。
[容量根拠](docs/research/track_a_h4_input_generation_stage_storage_review.md)と[最終承認資料入口](artifacts/resource_applicability/track_a_h4_input_generation_storage_review/2026-10-06/README.md)を参照する。
`H4_INPUT_GENERATION_STAGE_STORAGE_CONFIRMED_AWAITING_FINAL_APPROVAL`で公開後STOP。
新科学/追加transpile/taskset/worker/GPU/共有環境・他job変更0。旧10GiB案を含む以下は当時の履歴。

## 2026-10-06 H4入力生成 CPU/launch最終案・利用者承認未取得

提案CPU[3,5,6,7,8,9]、6 worker、異なる6 physical core・NUMA0。約3秒の受動負荷sampleで各core busy0%。
source19 pathsとplan v2 bytes/source_rootは不変。authは候補CPU集合だけ、reviewはauth digestだけ変更しapproved=falseを保持。
own新規runだけにtaskset maskを指定する未実行commandを用意した。CPU許可/専有予約/独立review/明示launchは未取得。
memory/context read-only確認は成功。filesystem空き約3.717GiBは総上限10GiB全量確保案に未達で、launch容量条件は未解決。
12 metadata gate tests PASS、fail/error/skip0。観測01/02の失敗logと03の容量未解決記録を保持し、追加transpile0/旧28件不変。
[提案資料](docs/research/track_a_h4_input_generation_cpu_launch_proposal.md)と[bundle・一括承認判断](artifacts/resource_applicability/track_a_h4_geometry_input_generation_cpu_launch_proposal/2026-10-06/README.md)を参照する。
`H4_INPUT_GENERATION_CPU_LAUNCH_PROPOSAL_FROZEN_AWAITING_APPROVAL`で公開後STOP。taskset/科学/worker/GPU/共有環境・他job変更0。

## 2026-10-06 H4入力生成 resource observer修正・未承認草案再固定

真のv2 hierarchy rootをnamespace/mount/所属から判定し、rootの非root memory interface要求を修正した。
全可視非root祖先の制限・pressure/OOMは保持し、非root欠測/不明namespace/hidden mountはSTOP、host-only fallbackなし。
observer33 zero-science tests PASS、実read-only観測成功。準備観測available約981.826GiB、PSI/OOM0はlaunch成立ではない。
production変更はresources observerとgateのnew audit pathだけ。科学/並列/seed/compiler source15件は不変。
[実装資料](docs/research/track_a_h4_geometry_resource_observer_fix.md)と[source bundle](artifacts/resource_applicability/track_a_h4_geometry_resource_observer_fix/2026-10-06/README.md)、
[new認可草案v2](artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06-v2/README.md)を参照する。source固定→別草案commit、binding検査結果はv2へ記録。
requested workers6、allowed_cpus=[]、approved=false。CPU/launch context・独立最終review・明示launch未解決、実行準備完了とはしない。
`H4_INPUT_GENERATION_RESOURCE_FIX_FROZEN_AWAITING_REVIEW`で公開後STOP。科学/追加transpile/GPU/本番起動/共有環境・他job変更0。
旧bundle/科学証拠/原稿/Track Bと旧監査履歴を保存し、系列transpile28/64を維持する。

## 2026-10-06 H4 geometry 入力生成専用認可草案・実行未承認

凍結science source6a121725（17 Python＋親2件）を変えず、入力生成source-bound planとresult-prior認可草案を追加した。
requested workers6、inputs/freeze digestはnull、218 templatesを機械転記。reviewはapproved=false、allowed_cpus=[]。
CPU許可は利用者指示で未確定のまま。現在process CPU0–255を許可とみなさず、launch contextとmemory観測は未解決。
既存observerはroot cgroup memory.max欠落で停止し、実行準備完了とはしない。source/共有設定を緩和しない。
新57 zero-science gate tests PASS、fail/error/skip0。合格経路はメモリ内模擬承認だけ、追加transpile0・旧累積28/64不変。
[実装・停止条件](docs/research/track_a_h4_geometry_input_generation_authorization_draft.md)と[bundle・最終レビュー入口](artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06/README.md)を参照する。
`H4_INPUT_GENERATION_AUTHORIZATION_DRAFT_FROZEN_AWAITING_REVIEW`で公開後STOP。科学/GPU/本番起動/共有環境・他job変更0。
有効execution authorization0、final review/利用者の明示launch未実施。入力生成・本計算・signal/compile認可・H6/Track Bへ進まない。

## 2026-10-06 H4 geometry compile並列source再固定・科学未実行

compileの逐次waitをadmitted worker数以下のbounded投入・回収へ変更した。処理中ownerを追跡し、
COMPLETEとidentity/digest検査後だけ再利用する。trajectory/axis順・weight、科学scope・seed/compilerは不変。
[実装資料](docs/research/track_a_h4_geometry_parallel_source_implementation.md)と[new bundle](artifacts/resource_applicability/track_a_h4_geometry_parallel_source/2026-10-06/README.md)を現在の入口とする。
既存94＋並列回帰17＝111 synthetic tests PASS、fail/error/skip0。今回transpile3、旧25＋新3＝28/64。
fake futures/mock workersだけで制御を検査し、実worker/production性能は未検証。旧bundle/audit/契約・保存証拠は不変。
SOURCEとsourceを変更しないREVIEWの2 commitを分ける。分子アクセス/科学処理/GPU/本番起動/認可発行/共有環境・他job変更0。
`H4_GEOMETRY_PARALLEL_SOURCE_FROZEN_AWAITING_REVIEW`で公開後STOP。入力生成plan/auth作成・本計算・H6/Track Bへ進まない。

## 2026-10-06 H4 geometry server-native source固定・科学未実行

利用者の新指示で契約v2 D1〜D4を実装条件へ採用し、旧未承認履歴を保存した。
新namespace `src/trottertracks/resource_applicability/h4_geometry/`、二つのfuture runner、専用synthetic testsの入口は
[実装資料](docs/research/track_a_h4_geometry_server_native_source_implementation.md)と
[bundle](artifacts/resource_applicability/track_a_h4_geometry_source/2026-10-06/README.md)。
最終94 tests pass、fail/error/skip0。失敗・再検査込みsynthetic transpile25/64、旧benchmark128再実行0。
SOURCE_COMMITとsourceを変えないREVIEW_BUNDLE_COMMITを分離し、actual blob/hashは別監査で固定する。
旧247 source・v1/v2 bundle・準備25 files・保存6 JSONは不変。分子入力/科学処理/本番runner launch/GPU/環境・他job変更/認可発行0。
`H4_GEOMETRY_SOURCE_FROZEN_AWAITING_REVIEW`で公開後STOP。別入力生成authorizationの作成へ進めるかをレビューし、今回は発行・実行しない。
以下は各milestone当時の履歴。


## 2026-10-06 H4 geometry契約v2・レビュー待ちSTOP

現在の入口は[契約v2 bundle](artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06-v2/README.md)。v1 commit `7c1a3d43f61c5501a9e79206b7c60933f94b1077`を保存し、
D1〜D4を具体的な採用案、memory admissionを8+8w+16 GiB、認可を入力生成→freeze STOP→別signal/compile認可へ分離した。
H4 linear/STO-3G/DF rank12、6距離・218 template・32 paired trajectories・74,784上限は不変。8 system＋ancilla1(index8)、合計9 qubits。
[pure JSON validator](artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06-v2/contract_validator_v2.py)と[専用合成検査](artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06-v2/run_contract_tests_v2.py)はreview用で、science source/runnerではない。
新規320件pass（fail/skip0）、旧129件は保存・runner再実行0。旧v1 manifestはbase blobで照合し書き換えない。
D1〜D4レビュー承認は未解決、science/source port/input generation/next stage認可false、plan未seal、mandatory STOP。
今回の公開指示は軽量契約bundleと関連文書だけのcommit/non-force push。以下は各段階当時の履歴。

## Track A 原稿保留とH4 geometry拡張の準備

利用者の新しい指示により原稿作成・投稿先検討はいったん保留。
[H4 geometryと要求精度の全候補map設計案](docs/research/track_a_geometry_precision_extension_proposal_v0.md)と
[別JSON](docs/research/track_a_geometry_precision_extension_proposal_v0.json)を現在の準備入口とする。
218 template、1点12,464 wrapper、追加6点74,784 wrapperの案。距離・host・worker・新source/authorizationは未固定。
サーバーCPUを第一候補に同一synthetic fixtureで比較する設計で、性能確認・本計算はまだ実行していない。
[サーバー側への準備指示](docs/research/gpu_server_track_a_h4_geometry_resource_preparation_prompt.md)は既存server環境を優先し、
環境確認・CPU benchmark・契約草案までを依頼する。旧local version一致を強制せず、compiler差を別layerにする。
原稿v0.1/v0.2・図・旧結果/STOPは保存し、科学計算STOP。H6/H8・追加96・Track B統合は別判断。
以下は各milestone当時の履歴であり、旧認可を新条件へ使い回さない。

## Track A 通し原稿v0.1・主図完成（2026-10-05）

最新は[原稿・補足・監査の入口](docs/manuscripts/README.md)。保存証拠だけから4主図と補足図1、
日本語通し原稿、再現性付録、claim audit、投稿可能性review依頼を作成した。
[表示script](scripts/resource_applicability/build_track_a_manuscript_figures.py)と
[bundle照合script](scripts/resource_applicability/verify_track_a_manuscript_bundle.py)、専用21 synthetic/local testsで追跡する。
9入力のcommit blob/hashと22生成assetを照合。新科学計算0、旧result/status/manifest・Track B不変。
原稿作成時点はlocal uncommitted。利用者の指示で原稿bundleのみをcommit/pushする。
投稿可判定は未実施。次は完成原稿review、科学計算STOP。以下は各milestone当時の履歴。

## Track A PM-2後review採用 原稿化へ

現在の入口は[主張・証拠対応表](docs/research/track_a_post_pm2_claim_evidence_map.md)と
[原稿・主要4図の設計](docs/research/track_a_post_pm2_manuscript_design.md)。
PM-2結果commit 5a1adffad780f0ec4272f5e8bb94713f9ff0f2bcを根拠に、固定DF・二次PF・
canonical finite-RTE・所定shot規則の有限signal事例研究として閉じる方針を採用した。
近接discard反証、B2内の精度依存、元5構成のtransferを中心に、既存結果・POSTHOC・解釈を分ける。
今回は対応表と図表設計まで。図生成・通し原稿は未実施、新科学計算0、Track B変更0。
旧result/status/manifestは不変で、科学計算のmandatory STOPを維持する。以下は各stage当時の履歴である。

## Track A PM-2保存値解析完了 研究方針review待ちSTOP

[PM-2結果と照合](docs/pr2_pm2_precision_resource_result_validation.md)から
`artifacts/resource_applicability/pr2_pm2_precision_analysis/2026-10-05/`の結果・監査へ辿れる。
source `324435d`不変、ε=0.05の223候補を再現し、302点・67,346行を照合した。
developmentのprimary点最小は精度要求によりB2 q4/q2/q1へ変わる。M2元5構成は別domainで、厳密winnerは認定しない。
pre/post62 local tests passed、新しい科学計算0。利用者指示でresult commitへ収録するPOSTHOC local evidence。
`PM2_PRECISION_RESOURCE_MAP_COMPLETE_AWAITING_REVIEW`、mandatory STOP、研究判断null、次段未認可。
以下の準備・source固定・旧stageの記述はmilestone当時の履歴である。

## Track A PM-2解析source固定 本解析は明示指示待ち

[保存値解析実装](docs/research/pr2_pm2_precision_analysis_implementation.md)のsourceを
`324435d77b6642dbd44e8d1f178420daf62e77ed`で固定した。stdlib-only解析module、future runner、専用synthetic testsの一組で読む。
`PM2_ANALYSIS_SOURCE_FROZEN_AWAITING_USER_LAUNCH`、62 local synthetic tests passed、fail/skip0。
実データのreference gate・精度走査・順位・P envelope、新signal/sampling/compileは未実行。
source/準備blobを照合した[固定後監査](artifacts/resource_applicability/pr2_pm2_precision_implementation/2026-10-05/source_freeze_v1.json)を参照する。
以下の準備・旧stage記述は各milestone当時の履歴として保持する。

## Track A PM-2契約準備完了 解析は未認可

[PM-2精度と資源境界契約](docs/research/pr2_pm2_precision_resource_contract_v1.md)を固定した。
`src/trottertracks/resource_applicability/pm2_precision_contract.py`、
`scripts/resource_applicability/run_pr2_pm2_precision_contract.py / run_pr2_pm2_preparation_tests.py`、
`tests/tracks/resource_applicability/test_pm2_precision_contract.py`、
`artifacts/resource_applicability/pr2_pm2_precision_preparation/2026-10-05/`を一組として読む。
全218 development候補とM2元5構成を別集合にし、POSTHOC、ε=0.005〜0.1、α_axis=0.025、軸別費用、共通P>=0を固定する。
保存JSON4件のidentity・field coverageだけを検査した。precision sweep/ranking/P envelopeは0。
`PM2_PRECISION_CONTRACT_FROZEN_ANALYSIS_NOT_AUTHORIZED`でSTOP。以下はmilestone当時の履歴として保持する。

## Track A PM-1実行完了：研究方針review待ちSTOP

最終承認・利用者の明示launch後、[PM-1結果照合](docs/pr2_pm1_discard_result_validation.md)を完了した。
H4 linear 1.00 Å、STO-3G、DF rank12、8 qubits、T=0.8、B0 rank4/5 × q=1/2/4/8、
delta=0.8/0.4/0.2/0.1、r=K=0の8構成が全件accuracy適格。8 signals/16 wrappers、CPU1、wall141.614秒。
最小の新B0 rank5・q1はG_RZ=229,718,060、保存B2 rank3・q1・r4・K2のpoint値の1.75659倍、
旧B0 rank6・q1より9.34%低い。これは固定development比較でありgeneral method optimumではない。
source134・plan・authorization・runner manifestを照合し、pre/post201 local tests passed、fail/skip0。
random/held-out/GPU/quantum shots/retry0。結果・監査は
`artifacts/resource_applicability/pr2_pm1_discard_execution/2026-10-04/`のlocal evidenceを利用者指示でresult commitへ収録する。
**statusはPM1_DISCARD_MAP_COMPLETE_AWAITING_REVIEW。mandatory STOP、研究判断null、次段未認可。**
旧draft/finalization/準備の記述はmilestone当時の履歴として保存し、現在地はこの節を優先する。

最終更新：2026-10-05

## Track A post-M2レビュー資料

Track Aの最新解析は[PM-0 POSTHOC証拠帰属](docs/research/pr2_post_m2_evidence_attribution.md)。
`src/trottertracks/resource_applicability/pm0_evidence_attribution.py`、
`scripts/resource_applicability/run_pr2_post_m2_evidence_attribution.py`、
`tests/tracks/resource_applicability/test_pm0_evidence_attribution.py`、
`artifacts/resource_applicability/pr2_post_m2_evidence_attribution/2026-10-04/`を一組として読む。
保存JSON/sourceだけを再集計し、source-bound科学コードや旧M1/M2証拠は変更しない。PM-1以降は未認可。

Track Aの次段準備は[PM-1近接discard契約](docs/research/pr2_pm1_nearby_discard_contract_v1.md)。
`src/trottertracks/resource_applicability/pm1_discard_contract.py / pm1_discard_execution.py`、
`scripts/resource_applicability/run_pr2_pm1_discard_contract.py / run_pr2_pm1_discard.py`、
`tests/tracks/resource_applicability/test_pm1_discard.py`、
`artifacts/resource_applicability/pr2_pm1_discard_preparation/2026-10-04/`を一組とする。
sourceはcommit `fd7552e`、sealed planと134 source blobの照合まで。本計算・authorizationは未認可。
[GPTへの準備bundleレビュー依頼](docs/research/pr2_pm1_preparation_external_review_request_fd7552e.md)に
固定identity、読む資料、停止条件をまとめた。利用者の別指示によりレビュー資料のみcommit/pushする。

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


## H4 geometry 契約準備 v1（2026-10-06・local未commit）

[準備bundle](artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06/README.md)は契約schema・zero-compute plan・pure JSON validatorと129合成検査の入口。
6距離、218 template/点、74,784 wrapper、最大12 workersを固定し、生成/seed/memory/wall/outputはreview待ち。
science source/runnerの追加ではなく、旧公開draft・source・科学結果は不変。本計算・port・commit/pushは未認可、STOP。


## H4候補間compile投入・12 worker明示再実行

利用者の増員再実行指示により[run03 source・認可・検査記録](docs/research/track_a_h4_cross_candidate_run03.md)を追加した。候補内2回路の完了待ちで4 workerがidleとなる問題を、候補間bounded queueで修正。旧run02はworker failure STOPで全証跡を保持し、旧6入力を再生成せず利用する。49人工job/metadata testsはlocal PASSで科学的結果ではない。12 workerのfresh CPU/memory/容量/hash検査後だけ一度起動しMAP_COMPLETE_STOP。旧累積bytes/wall/actual invocationsを引継ぎ、科学条件・compiler・上限は不変。


## H4 run04：identity hash分割・5秒監視維持

[run04固定sourceと再実行binding](docs/research/track_a_h4_streaming_monitor_run04.md)を追加した。run03はmonitor STOPで旧全証跡を保持。人工JSONで旧一括encoderのmonitor interval/freshnessを再現し、新streaming encoderは同じdigestで全監視PASS。65 local tests PASSは実装証拠で科学結果ではない。旧6入力を再生成せず、bytes/wall/7 consumed invocationsをcarryし、CPU12件・12 workerのfresh検査後に一度実行してMAP_COMPLETE_STOP。


## H4 run05：identity hash分割・5秒監視維持

[run05固定sourceと再実行binding](docs/research/track_a_h4_lazy_identity_run05.md)を追加した。run04はmonitor STOPで旧全証跡を保持。人工JSONで旧一括encoderのmonitor interval/freshnessを再現し、新streaming encoderは同じdigestで全監視PASS。75 local tests PASSは実装証拠で科学結果ではない。旧6入力を再生成せず、bytes/wall/12 consumed invocationsをcarryし、CPU12件・12 workerのfresh検査後に一度実行してMAP_COMPLETE_STOP。


## H4 新サーバー引継ぎ（2026-10-07）

[引継ぎ資料入口](docs/research/track_a_h4_new_server_handoff_20261007.md)に、固定sourceの取得手順、run05監視STOP、累積予算、入力・証拠の別転送、新hostの未承認事項をまとめた。今回の追加は軽量資料のみ。本計算・入力再生成・GPU・共有環境変更を開始しない。


## 2026-10-07 H4新host source取得・環境解決案

[H4新host source取得・環境解決案](docs/research/track_a_h4_new_server_preparation_20261007.md)。指定commit/source19照合済み、環境18 version/45 RECORD差の解決案でSTOP。source修正・新tests・production未実施、入力/停止証拠別送待ち、allowed_cpus=[]/approved=false。旧科学結果・原稿・Track Bは保持。


## 2026-10-07 H4新host A案 source/人工検証固定・本計算未認可

[監視修正・32人工tests](docs/research/track_a_h4_new_server_monitor_fix_a_20261007.md)。既存private venv不変で準備用A案採用。
旧source19は3変更/16不変、新module込み25 closure。256×256人工matrix＋9-qubitの旧/new byte/digest一致、独立observerのGIL/GC観測・I/O delay/EOF/所有・資源境界を検証。
SOURCE `b2a5ad89e8b39d72716f7ddb17d263bd0cdedb45`、production/追加transpile0、旧28/64・benchmark128保持。環境18 version差、旧45 raw-reference RECORD差保持・normalized22差を明記。入力6/freeze/runtime/control未受領。
observer AS256MiB/RSS64MiB/admission120.25GiB、容量5.5625GiB案は未承認。allowed_cpus=[]/approved=false/runtime_authorization=false/launch=null、STOP。科学成果/原稿/Track Bと旧資料を保持。

人工runner `scripts/resource_applicability/run_h4_monitor_fix_a_tests.py`、auditor `scripts/resource_applicability/audit_h4_monitor_fix_a.py`。実runnerを起動しない。


## 2026-10-08 H4本計算前準備・最終承認待ちSTOP

[統合入口・最終承認案](docs/research/track_a_h4_prelaunch_preparation_20261008.md)。新host schema/profile/observer/CPU/one-shot/累積budget bindingを整備。
SOURCE `ad57d1639133f7158cce58d767b8e0aa179bf044`、32 closure、57限定人工tests PASS。core quota read-only確認、worker12＋driver/observer各1別coreを提案。
carry20/165214360 bytes/5466.188392877579秒、残74764を保持。全74784 logicalを保証する最小actual cap+20→74804案は未承認。
候補環境はprivate venv不変、18 version/旧参照対raw45・normalized22差。追加output5GiB/301000 inodes、charge約8.29GiB、copy前は暫定6GiB。
入力6/freeze/native stop proof未受領、allowed_cpus=[]/approved=false/runtime_authorization=false/未seal、science/追加transpile/GPU/共有環境・他job変更0。明示承認・final review・launch前にSTOP。


## 2026-10-08 H4受領監査・独立最終review TECHNICAL FAIL

[統合入口](docs/research/track_a_h4_receipt_final_review_20261008.md)。origin79827016/SOURCEad57d163照合、source32不変、read-only live profile/資源/quotaを確認。
凍結NPZ6/freeze/native stop/controlは未受領、producer bytes/SHA manifestを含む最小転送手順を具体化。科学array読込/再生成0。
独立reviewはpidfd送信時ESRCH競合で後続cleanupが中断するP1を純mock再現しTECHNICAL FAIL。57人工PASSはこの競合を覆わない。
未受領NOT_EVALUABLE、CPU/env/observer/74804案のUNAPPROVEDと技術FAILを区別。sourceは修正せず再sealなし、flags false、carry20/165214360/5466.188392877579を保持し本計算STOP。

## 2026-10-09 H4 cleanup ESRCH修正・独立再review PASS

[新SOURCE・独立再review・残る承認条件](docs/research/track_a_h4_cleanup_esrch_fix_20261009.md)。旧992c09d6から独立worktreeでP1を修正し、SOURCE `6bd1ba01cd71ec3e2071082963c9f07478dada9a`・closure33を固定。
両pidfd送信経路はESRCHだけ既退出扱いで後続cleanupを継続し、他の送信error・所有検証を保持する。
限定人工39件PASS、別担当13純mock件PASSとbinding照合でP1_SCOPE_TECHNICAL_PASS。wait/reap/FD/pipe/first STOP理由保持まで確認。
新test `tests/tracks/resource_applicability/test_h4_cleanup_esrch.py` と既存人工runnerで追跡し、source/profile/plan/auth/reviewを新SHAへ再結合した。
入力0/6・freeze/native停止証拠未受領はNOT_EVALUABLE、未seal。環境/CPU/observer/74804案は未承認、carry20/165214360 bytes/5466.188392877579秒・現actual cap74784不変。
approved=false、runtime_authorization=false、allowed_cpus=[]。追加transpile/科学actual/本番起動0。旧資料を保存して本計算STOP。

## 2026-10-09 H4全byte受領・carry合格・native終端proof待ち

[受領・binding・最終承認案](docs/research/track_a_h4_byte_receipt_binding_20261009.md)。packet86691840B/全2150files/既知9SHAをbyte-only照合しPASS、NPZ6/freeze受領完了。
SOURCE `6bd1ba01cd71ec3e2071082963c9f07478dada9a`・closure33不変。ledger chain/cumulative journalからcarry20/165214360 bytes/5466.188392877579秒を保持、現cap74784・残74764。
run05 log/exact旧sourceからworker cleanup到達は推認可能だが、driver/12 workers停止後identity/残存0 native proofが不足。古いrun02停止監査をrun05proofへ流用しない。
input/profile/source/output/carryとv6 plan/auth/reviewを結合し、control82件のbasename・単一link・sender manifest mappingを照合。source条件は緩めていない。
sealed=false/approved=false/runtime_authorization=false/allowed_cpus=[]。環境/CPU/observer/累積74804案/一度のmap launchは未承認。科学array読込/新科学actual/追加transpile/共有設定変更0でSTOP。

## 2026-10-09 H4追加native停止proof合格・technical再seal

[再seal・独立最終整合review・一括承認案](docs/research/track_a_h4_native_proof_seal_20261009.md)。追加JSON17521B/SHA一致、旧host/run05の13identity・2回残存0・元3証拠hashを照合して現在のnative停止条件PASS。
過去のexit code/正確な終了・reap時刻/原boot IDは未記録のままnull。今回の観測で補完せず、連続監視や歴史cleanup順の証明とも扱わない。
SOURCE `6bd1ba01cd71ec3e2071082963c9f07478dada9a`・closure33不変、profile/input/carry/control83件の固定validator合格でplan再seal、sealed=true。
approved=false/runtime_authorization=false/allowed_cpus=[]。carry20/165214360 bytes/5466.188392877579秒、現cap74784・残74764を保持。
候補environment/compiler・CPU・observer・累積actual74804案・一度のmap launchは未承認。承認による最終artifact/digest再結合とfresh resource/CPU/fs/inode/quota gateをlaunch前に確認。
科学array読込/新science actual/追加transpile/source変更/共有設定変更/本計算0でSTOP。

## 2026-10-09 H4一度のmap実行を利用者承認・最終artifact固定

[実行認可・直前gate・起動報告入口](docs/research/track_a_h4_authorized_launch_20261009.md)。利用者の明示認可で候補environment/compiler採用、worker12 CPUs2/4–6/8–15・driver16・observer18、observerAS256MiB/RSS64MiB/admission120.25GiBを認可。
carry20/165214360 bytes/5466.188392877579秒を保持し、累積actualだけ74804へ+20改定。SOURCE6bd1ba01・science/compiler/options/他caps不変。
sealed/approved/runtime_authorization=true、allowed_cpusはexact14role集合。独立v8 review PASS。artifact commit後fresh CPU/memory/PSI/OOM/FS/inode/quotaとSOURCE/profile/input/carry/unusedrootを確認しPASSなら追加承認なし一度起動。
既存proof/回帰/benchmark再実行0、追加準備campaign/transpile0。oldpartial/cache/GPU/共有環境・venv・他job変更なし、完了またはfail-closed STOP後終了・retry/次stageなし。
これは認可artifact固定時点のsnapshot。実起動/PID/状態は入口への追記・外部runtime receiptで別記録する。


## 2026-10-09 H4一度起動・Matplotlib run外設定write拒否STOP

[実結果](docs/research/track_a_h4_authorized_launch_20261009.md)。immutable認可/fresh gate PASS後driver2740551/observer/12 workersを一度起動したが、第一candidate準備のOpenFermion→Cirq→Matplotlib config mkdirが保護guardに拒否されFAIL_CLOSED_STOP。実exit1、14 own identities残存0。newtranspile0/累積actual20、charged4428938712 bytes/wall5472.345380863175秒は返却せず維持。retry/入力再生成/次stage/GPU/共有設定変更なし。

## 2026-10-09 H4 library cache保存先修正・再実行予算不合格

[修正・再実行条件](docs/research/track_a_h4_library_cache_fix_20261009.md)。前回のOpenFermion→Matplotlib mkdir拒否を、homeの新private library cacheへprocess限定MPLCONFIGDIRを結合して修正。
driver/workerとも既存directoryのEEXIST probe以外のcache writeを拒否、29816BのSHA固定。47限定回帰＋3 import case PASS、科学array/transpile/実worker/affinity/GPU0。
SOURCE `b8b3ce6e8c98f1ec0419a7af79c5d7c5f3a3b9bb`、36 closure、science/compiler/options不変。前回費用を返却せずcarry20/4428938712B/5472.345380863175sへ結合。
累積worst charge13165893832B=12.261694GiB>承認10GiBで未seal/approved=false/runtime_authorization=false、再起動0。13GiBは未承認proposalのみ。
既承認environment/CPU/observer/actual74804を保持。cap改定・新SOURCE/gate binding/review・fresh gate後の一度再実行が残る。
private homeは共有systemと区別し、旧run/one-shot/失敗証拠・全予約課金を保持する。

## 2026-10-09 H4軽量高速化・限定同等性確認

実装：[signal](src/trottertracks/resource_applicability/h4_geometry/signal.py)・[ledger](src/trottertracks/resource_applicability/h4_geometry/ledger.py)。限定[runner](scripts/resource_applicability/run_h4_lightweight_speedup_tests.py)・[tests](tests/tracks/resource_applicability/test_h4_lightweight_speedup.py)・[bundle](artifacts/resource_applicability/track_a_h4_lightweight_speedup/2026-10-09/README.md)。

[変更・限定検証・binding](docs/research/track_a_h4_lightweight_speedup_20261009.md)。driverの距離内共通準備を再利用し、one/DF block呼出を静的13×218→13、全prepareを218→10種類に削減。ledger deltaは変更entry/reservation各最大1だけを参照し、全件走査・saved-historyコピーを除いた。
SOURCE `4d2d1492fc23d0736c305533d78967cc1db8a7c8`、closure38。限定48人工PASS、全218prep/代表8wrapper+dense256case1/代表4signal/ledger13fileの旧new bytes・digest一致。12workersはmock、単一test process内部thread1、science array/transpile/GPU/affinity/production0。
実Gaussian/旧compiler output/実速度・H4本体成功は未検証。monitor/caps/compiler/science/carry不変、旧partial/cacheと混合しない。
carry20/4428938712B/5472.345380863175s、actual74804既承認、worst charge12.261694GiB>承認10GiBは残る。未seal/approved=false/runtime_authorization=false、absolute_launch_command=null、本計算0。

## 2026-10-09 H4利用者が13GiB累積charge・一度の再実行を明示認可

[認可・source・一度の起動入口](docs/research/track_a_h4_approved_relaunch_20261009.md)。利用者の「これについては問題ないので再実行して」を、既存13GiB cumulative charge案と一度のmap再実行の承認として反映。
SOURCE `a7b617600cd7063f7870f2059d5694ef00283f0e`/closure39、output改定schema/gate/実OutputBudget capとmarginのみ変更。legacy10GiB default・科学/compiler/その他caps・prepare再利用/ledger保存は維持。
限定22pure gate PASS、旧48speedup/library/cleanup/native証拠campaign再実行なし。carry20/4428938712B/5472.345380863175s返却なし、actual74804・新残74784。
worst13165893832B <= 新cap13958643712B、余裕792749880B。sealed/approved/runtime_authorization=trueの認可artifactへ再結合。
独立review/artifact固定後fresh SOURCE/profile/input/carry・CPU/memory/PSI/OOM/fs/block/inode/quota/unusedroot/one-shot合格時にそのまま一度起動。実run状態はruntime証跡へ記録。
既承認12workers CPUs2/4–6/8–15、driver16/observer18、thread1・observerAS256MiB/RSS64MiB/admission120.25GiB保持。自動retry/入力再生成/旧partial/cache/次stage/GPU/共有設定変更なし。
