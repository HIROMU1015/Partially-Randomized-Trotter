# 研究ノート

## 2026-10-06 H4 signal/compile INPUT_BOUND草案・容量不足STOP

run02の6凍結入力へplanを結合し、source19/218 templates/compiler/run ID/outputを不変に保った。
stage必要3.5 GiB/560000 inodesに対しavailable3.419376 GiB、約82.6 MiB不足。CPU候補6 core/memory/quota観測はPASS。
全72h監視・74784 records・149569 ledger deltas・全8KiB worker logs・1308 signal files・journal/temp/directory余裕を含む。
CPU [3,5,6,7,8,9]・6 worker・own-run mask0x3e8は次段の提案、review=false、利用者の別stage承認/明示launch未取得。
signal/seed/sampling/build/compile/transpile/taskset/worker/GPU/共有環境・他job変更0、入力再生成0。
[次段scope・容量](../track_a_h4_signal_compile_plan_review.md)と[資料入口・承認対象](../../../artifacts/resource_applicability/track_a_h4_signal_compile_plan_review/2026-10-06/README.md)を参照する。
`H4_SIGNAL_COMPILE_PLAN_PREPARED_STORAGE_BLOCKED_STOP`。容量と別認可成立後もfresh検査不合格なら起動せず、map後MAP_COMPLETE_STOP。
scientific runtimeはcommitしない。旧10GiB案/旧source/旧bundle/入力生成完了と以下の履歴は保持する。

## 2026-10-06 H4入力生成run02・6入力freeze完了STOP

利用者の明示再実行指示でworker bootstrapのstdlib signal shadowを-Pで修正し、run01を保存してrun02を別固定した。
CPU [3,5,6,7,8,9]・6 workers・own-run限定、fresh resource/容量/quota検査PASS。
H4 linear/STO-3G/DF12の追加6距離入力を一度生成しfreeze完了。NPZ6 bytes SHAを照合し、own driver/worker残存0。
sourceは049e69919af16ad29a67a217dc7a407d6b1754a6。科学/seed/compiler/resource条件は不変、source19変更はgates/worker起動だけ。
`INPUTS_FROZEN_STOP`、next_stage_authorized=false、mandatory_stop=true。signal/compile/GPU/追加transpile0、共有環境・他job変更0。
[修正・完了scope](../track_a_h4_worker_bootstrap_run02.md)と[完了報告](../../../artifacts/resource_applicability/track_a_h4_worker_bootstrap_run02/2026-10-06/COMPLETION_REPORT_v1.md)を参照する。scientific runtimeはcommitしない。以下は各時点の履歴。

## 2026-10-06 H4入力生成stage容量確認・最終承認待ち

容量準備の判定は「足りる」。6入力生成→freeze→STOPの必要量3GiB/260000 inodesに対し、
2026-10-06 16:58:53 JSTのnonroot available約3.615GiB、225817022 inodes、user/group/project quota非有効をread-only確認。
32保存配列/NPY・ZIP overhead/64MiB IPC上限/temp-final/72h監視259202 files/journal/metadata余裕を含む。
全campaign10GiBはcharge capとして維持し、旧全量空き確保案を履歴に保存した上でstage-specific補足を追加した。
source19/plan/auth/approved=false reviewはbyte-identical、CPU [3,5,6,7,8,9]・6 worker・own-run mask0x3e8は未承認。
CPU使用許可/独立最終review/明示launch/fresh CPU・memory・pressure/OOM・容量検査が残る。signal/compile容量は別認可。
[容量根拠](../track_a_h4_input_generation_stage_storage_review.md)と[最終承認資料入口](../../../artifacts/resource_applicability/track_a_h4_input_generation_storage_review/2026-10-06/README.md)を参照する。
`H4_INPUT_GENERATION_STAGE_STORAGE_CONFIRMED_AWAITING_FINAL_APPROVAL`で公開後STOP。
新科学/追加transpile/taskset/worker/GPU/共有環境・他job変更0。旧10GiB案を含む以下は当時の履歴。

## 2026-10-06 H4入力生成 CPU/launch最終案・利用者承認未取得

提案CPU[3,5,6,7,8,9]、6 worker、異なる6 physical core・NUMA0。約3秒の受動負荷sampleで各core busy0%。
source19 pathsとplan v2 bytes/source_rootは不変。authは候補CPU集合だけ、reviewはauth digestだけ変更しapproved=falseを保持。
own新規runだけにtaskset maskを指定する未実行commandを用意した。CPU許可/専有予約/独立review/明示launchは未取得。
memory/context read-only確認は成功。filesystem空き約3.717GiBは総上限10GiB全量確保案に未達で、launch容量条件は未解決。
12 metadata gate tests PASS、fail/error/skip0。観測01/02の失敗logと03の容量未解決記録を保持し、追加transpile0/旧28件不変。
[提案資料](../track_a_h4_input_generation_cpu_launch_proposal.md)と[bundle・一括承認判断](../../../artifacts/resource_applicability/track_a_h4_geometry_input_generation_cpu_launch_proposal/2026-10-06/README.md)を参照する。
`H4_INPUT_GENERATION_CPU_LAUNCH_PROPOSAL_FROZEN_AWAITING_APPROVAL`で公開後STOP。taskset/科学/worker/GPU/共有環境・他job変更0。

## 2026-10-06 H4入力生成 resource observer修正・未承認草案再固定

真のv2 hierarchy rootをnamespace/mount/所属から判定し、rootの非root memory interface要求を修正した。
全可視非root祖先の制限・pressure/OOMは保持し、非root欠測/不明namespace/hidden mountはSTOP、host-only fallbackなし。
observer33 zero-science tests PASS、実read-only観測成功。準備観測available約981.826GiB、PSI/OOM0はlaunch成立ではない。
production変更はresources observerとgateのnew audit pathだけ。科学/並列/seed/compiler source15件は不変。
[実装資料](../track_a_h4_geometry_resource_observer_fix.md)と[source bundle](../../../artifacts/resource_applicability/track_a_h4_geometry_resource_observer_fix/2026-10-06/README.md)、
[new認可草案v2](../../../artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06-v2/README.md)を参照する。source固定→別草案commit、binding検査結果はv2へ記録。
requested workers6、allowed_cpus=[]、approved=false。CPU/launch context・独立最終review・明示launch未解決、実行準備完了とはしない。
`H4_INPUT_GENERATION_RESOURCE_FIX_FROZEN_AWAITING_REVIEW`で公開後STOP。科学/追加transpile/GPU/本番起動/共有環境・他job変更0。
旧bundle/科学証拠/原稿/Track Bと旧監査履歴を保存し、系列transpile28/64を維持する。

## 2026-10-06 H4 geometry 入力生成専用認可草案・実行未承認

凍結science source6a121725（17 Python＋親2件）を変えず、入力生成source-bound planとresult-prior認可草案を追加した。
requested workers6、inputs/freeze digestはnull、218 templatesを機械転記。reviewはapproved=false、allowed_cpus=[]。
CPU許可は利用者指示で未確定のまま。現在process CPU0–255を許可とみなさず、launch contextとmemory観測は未解決。
既存observerはroot cgroup memory.max欠落で停止し、実行準備完了とはしない。source/共有設定を緩和しない。
新57 zero-science gate tests PASS、fail/error/skip0。合格経路はメモリ内模擬承認だけ、追加transpile0・旧累積28/64不変。
[実装・停止条件](../track_a_h4_geometry_input_generation_authorization_draft.md)と[bundle・最終レビュー入口](../../../artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06/README.md)を参照する。
`H4_INPUT_GENERATION_AUTHORIZATION_DRAFT_FROZEN_AWAITING_REVIEW`で公開後STOP。科学/GPU/本番起動/共有環境・他job変更0。
有効execution authorization0、final review/利用者の明示launch未実施。入力生成・本計算・signal/compile認可・H6/Track Bへ進まない。

## 2026-10-06 H4 geometry compile並列source再固定・科学未実行

compileの逐次waitをadmitted worker数以下のbounded投入・回収へ変更した。処理中ownerを追跡し、
COMPLETEとidentity/digest検査後だけ再利用する。trajectory/axis順・weight、科学scope・seed/compilerは不変。
[実装資料](../track_a_h4_geometry_parallel_source_implementation.md)と[new bundle](../../../artifacts/resource_applicability/track_a_h4_geometry_parallel_source/2026-10-06/README.md)を現在の入口とする。
既存94＋並列回帰17＝111 synthetic tests PASS、fail/error/skip0。今回transpile3、旧25＋新3＝28/64。
fake futures/mock workersだけで制御を検査し、実worker/production性能は未検証。旧bundle/audit/契約・保存証拠は不変。
SOURCEとsourceを変更しないREVIEWの2 commitを分ける。分子アクセス/科学処理/GPU/本番起動/認可発行/共有環境・他job変更0。
`H4_GEOMETRY_PARALLEL_SOURCE_FROZEN_AWAITING_REVIEW`で公開後STOP。入力生成plan/auth作成・本計算・H6/Track Bへ進まない。

## 2026-10-06 H4 geometry server-native source固定・科学未実行

利用者の新指示で契約v2 D1〜D4を実装条件へ採用し、旧未承認履歴を保存した。
新namespace `src/trottertracks/resource_applicability/h4_geometry/`、二つのfuture runner、専用synthetic testsの入口は
[実装資料](../track_a_h4_geometry_server_native_source_implementation.md)と
[bundle](../../../artifacts/resource_applicability/track_a_h4_geometry_source/2026-10-06/README.md)。
最終94 tests pass、fail/error/skip0。失敗・再検査込みsynthetic transpile25/64、旧benchmark128再実行0。
SOURCE_COMMITとsourceを変えないREVIEW_BUNDLE_COMMITを分離し、actual blob/hashは別監査で固定する。
旧247 source・v1/v2 bundle・準備25 files・保存6 JSONは不変。分子入力/科学処理/本番runner launch/GPU/環境・他job変更/認可発行0。
`H4_GEOMETRY_SOURCE_FROZEN_AWAITING_REVIEW`で公開後STOP。別入力生成authorizationの作成へ進めるかをレビューし、今回は発行・実行しない。
以下は各milestone当時の履歴。


## 2026-10-06 H4 geometry契約v2・レビュー待ちSTOP

現在の入口は[契約v2 bundle](../../../artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06-v2/README.md)。v1 commit `7c1a3d43f61c5501a9e79206b7c60933f94b1077`を保存し、
D1〜D4を具体的な採用案、memory admissionを8+8w+16 GiB、認可を入力生成→freeze STOP→別signal/compile認可へ分離した。
H4 linear/STO-3G/DF rank12、6距離・218 template・32 paired trajectories・74,784上限は不変。8 system＋ancilla1(index8)、合計9 qubits。
[pure JSON validator](../../../artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06-v2/contract_validator_v2.py)と[専用合成検査](../../../artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06-v2/run_contract_tests_v2.py)はreview用で、science source/runnerではない。
新規320件pass（fail/skip0）、旧129件は保存・runner再実行0。旧v1 manifestはbase blobで照合し書き換えない。
D1〜D4レビュー承認は未解決、science/source port/input generation/next stage認可false、plan未seal、mandatory STOP。
今回の公開指示は軽量契約bundleと関連文書だけのcommit/non-force push。以下は各段階当時の履歴。

このディレクトリには、研究実装を進めた時点の方針、判断、検証結果および
未解決事項を日付順に記録する。後から「なぜこの実装になったか」「その時点で
何が確認済みだったか」を、commitと検証コマンドまで含めて追跡できるようにする。

## 資料としての位置付け

研究ノートは時点ごとの作業記録であり、現行仕様の正本ではない。

- 現在の研究方針と評価条件：`docs/research/`の主資料
- API、回路scopeおよび数学的規約：各実装文書
- 再現可能性と保証status：`VALIDATION_STATUS.md`と
  `artifacts/validation_manifest.json`
- 実際の数値結果：fingerprintと生成条件を持つmachine-readable artifact

過去のノートと現行仕様が異なる場合は現行仕様を優先し、変更理由を新しい日付の
ノートに追記する。過去の記録を現在の理解に合わせて黙って書き換えない。

## 記録規則

1. ファイル名は日本時間の日付に対応する `YYYY-MM-DD.md` とする。
2. 同じ日に複数回更新する場合は、ファイル内に `HH:MM JST` の節を追加する。
3. 実装を記録するときは、基準commit、対象scope、採用方針と採用しなかった範囲を
   明記する。
4. 検証結果は実行コマンド、pass/fail/skip/warning数、既知の環境制約を記録する。
5. 結果には `確認済み`、`部分確認`、`未確認`、`blocked` のいずれかを付ける。
6. 科学的な結論は、対応するartifactとfingerprintがない限り、実装能力の確認と
   区別する。
7. 失敗や方針変更も削除せず、後続ノートから訂正内容を参照する。

新しい記録は[テンプレート](テンプレート.md)を複製して作成する。

## 時系列索引

| 日付 | 主題 | 基準commit | 到達点 | 次の主要課題 |
|---|---|---|---|---|
| [2026-10-05 PM-2結果](2026-10-05.md) | 保存値precision/resource map実行と照合 | source `324435d` / launch `bea4cf0` | 元223候補再現、302点・67,346行、pre/post62 tests、新しい科学計算0 | mandatory STOP、研究方針review。利用者指示でresult commitへ収録 |
| [2026-10-05 source固定](2026-10-05.md) | PM-2保存値解析実装・synthetic検査 | `324435d` | source6 blobs・準備9 files一致、62 synthetic tests。本解析/real reference gate未実行 | 別の明示解析指示待ち。終了後mandatory STOP |
| [2026-10-05追記](2026-10-05.md) | PM-1結果後方針を採用、PM-2精度・測定込み資源契約準備 | `194cc604` + local uncommitted preparation | 保存4 JSONのblob/hash、development218候補・M2元5構成別集合、専用21 local tests。precision解析0、新しい科学計算0 | 契約確認後、保存値解析の実装/source固定。実行は別指示、終了後STOP |
| [2026-10-05](2026-10-05.md) | Track A PM-1最終承認・一項目authorization確定 | `bf9eaee` | final_review_approvedのみtrue、source/plan不変、science-free gate PASS、201 local tests passed | 利用者の明示launch待ち。本計算0、実行後もmandatory STOP |
| [2026-10-03](2026-10-03.md) | PR-2 M1-B1 actual compile map検証 | source `33f436b` + local result | 12,448 wrapperと全checkpoint/cacheを再検査。B2 rank 3、q=1のactual frontierを確認し`CONTINUE_RESOURCE_STUDY` | result-prior held-out transfer reviewを別freeze。H4 1.30 Åは未開封 |
| [2026-09-30](2026-09-30.md) | PR-2 M1-A result-prior authorization | authorization commitで固定 | development-only最大212 signal、compile 0、held-out access 0を結果前固定 | M1-Aを一度実行し、limitedなら停止、clearならartifact freeze後に別M1-B authorization |
| [2026-09-29](2026-09-29.md) | PR-2 V4/S2完了とmatched-accuracy再設計 | `61bbaad` | 旧S2を保持し、M1前研究契約とzero-compute実装契約を固定。208候補・16-cell selector、専用test通過、M1科学計算未承認 | 独立review後、必要ならM1 execution authorizationを別freeze |
| [2026-09-27](2026-09-27.md) | FR-R1b完了と研究完成フェーズ移行 | `ecb7f4c`、`16d4482` + dirty worktree | R5不通過と`MECHANISM_ONLY_NO_PRACTICAL_GO`を維持。C1/C2を中核、C3を条件付き応用とする完成原稿契約を固定 | 新規計算を止め、定理単位の先行研究監査とT1--T4の証明へ進む |
| [2026-09-25](2026-09-25.md) | M06-F・A0・P-B/P-C/P-A停止点 | `3336f03` + dirty worktree | P-C tracking 16/16完了。stretch予測破れと診断不通過によりA/B/C全てcurrent scopeで停止 | P-Dを事前登録するかR3/R6/R8へ問いを再定義 |
| [2026-09-26](2026-09-26.md) | P-D S0契約・S1公平再最適化 | `9a494bd` + dirty worktree | B1b/B2/B4一致、Case C/D不成立。B1a上限依存でCase B＋undetermined、S1停止 | P-Dの研究価値・baseline設計を再検討。S2/H12/長RPEは保留 |
| [2026-09-24](2026-09-24.md) | M06-F all-r coherent opt2初期計算・解析 | `26aa95c` + dirty worktree | 36/36 cell完了。12 group中7通過、5 groupはRZ相対SE 2%基準でfresh-32待ち | 15 taskのfresh 32 trajectory拡張後にcoherent再最適化 |
| [2026-09-23](2026-09-23.md) | M06/L08、N07/P03、WP11限定判断統合 | `efa90e0` + dirty worktree | T4/T7を主軸、T1を範囲変更、T3を保留。次段はall-r coherent opt2再最適化 | opt2未測定$r=1,2,4,8,16$のCPU transpile・再最適化 |
| [2026-09-22](2026-09-22.md) | WP03、Gate S1、WP06-a/b、WP05-a/b/R、WP01-D/C07、G08/M08 | `efa90e0` + dirty worktree | M08の$q=16,32$直接holdoutはRZ最大3.286%で通過し、実測幅によるlocal再集計区間も分離。ただし25%移送区間は重なる | 主張範囲の見直しまたは外部条件での移送検証 |
| [2026-09-21](2026-09-21.md) | 研究方向screeningの実行gateとWP00/WP02/WP01-S/WP04 | `efa90e0` + dirty worktree | $L_D=0$をscreen out。WP04で公平な配分改善後の決定論endpoint差は4.96%へ縮み、5%・25%区間とも重なり未決定。主要因は$\beta$、次いで$\alpha$再配分 | WP03でPF係数選択感度を評価 |
| [2026-09-20](2026-09-20.md) | 4段RPE分枝復元、目標round診断、$\delta$/round別scheduleと中央RTE cost検証 | `2bf3116` + dirty worktree | H4の固定長round設定を棄却し、3個の$\delta$に行列検査を通るscheduleを構成。局所角度・$L=8,16,32$検証後の中央RTE proxyは0.02を全6指標で最小とした | $\delta=0.02$と0.01の制御付きpartial-$S_2$反復・Hadamard 1 shot costを検証 |
| [2026-09-18](2026-09-18.md) | 暫定配分を使った限定4段集計と新配分の短段失敗率 | `2bf3116` + dirty worktree | H4固定条件の$q=1,2,4,8$で1,572 shot、RZ数$3.2673961\times10^7$、8軸$\alpha$和0.05。$q=1,2,4$の厳密二項座標失敗率$2.2246\times10^{-4}$ | $q=8$物理信号、branch復元、必要全round・最終コストを別途検証 |
| [2026-09-01](2026-09-01.md) | RPE短段の信号・shot・cost接続、失敗率、$q=8$代理モデル、配分感度 | `2bf3116`に至る前のdirty worktree | 配分感度から$\beta=(0.02,0.02,0.36)$と重み付き$\alpha$を固定条件の暫定入力に選択 | 限定4段集計（2026-09-18に実施） |
| [2026-08-26](2026-08-26.md) | H5 connected-cluster系サイズ検証と回路cost modelの区切り | `e07a5e6` + dirty worktree | H5、rank 9、$L_D=4$、$K=2$、$L=4,6,8$でpaired K1--K3最大1.665%。独立calibration/holdoutは最大3.776%、予測半幅1.459%で5%/2%基準を通過 | cost providerをRPE shot・誤差/失敗確率配分へ接続。新compiler・$L>8$・不通過条件だけ追加holdout |
| [2026-08-24](2026-08-24.md) | 階層compiled-cost model、$K=2$次数条件付き再検証、connected-cluster運用推定と軽量化 | `e07a5e6` | 固定DF snapshotの$L=4,6,8$ holdoutでK1--K3運用推定は全metric最大2.936%。点誤差5%内だが95%診断5.724%の留保。calibration/prediction/transfer分離と厳密key cacheを実装 | 別$L_D$・short-step・compiler/coupling条件への移送と角度不変性検証 |
| [2026-08-25](2026-08-25.md) | 複数order-2、独立K4、controlled $q=8$の追加・follow-up batch | dirty worktree | paired複数order-2は最大1.679%。$L_D=6$のK1--K4 paired $L=8$は4.008%。controlled $q=8$は0.0529%。全job完走・validator通過 | 系サイズ方向の独立holdoutで運用規則を確認 |
| [2026-08-23](2026-08-23.md) | ランダム回路加法モデルとRTE境界補正の高統計検証 | `e07a5e6` | 1000標本・独立2 seedでcount/sizeのpair-only残差を確認。same/different二分類が別seed pair holdoutを最大0.849%で予測 | count/sizeの$\mu_3$または$L=8$、$L_D,K$、controlled・compiler条件のholdout検証 |
| [2026-08-19](2026-08-19.md) | 論文Eq. (D6)によるPF摂動係数の再検証 | `8418192` | H4全$L_D$とH2--H5の支配位相比較を通過し、H6のD6係数をstate-actionで算出 | GPU経路をH8/H10で確認し、H12の候補$L_D$ごとにD6係数を決定 |
| [2026-08-18](2026-08-18.md) | finite-RTEとPF・摂動・QPE分枝誤差の検証 | `8fdc6b3` | H4全$L_D$の単一位相条件、H2--H5のdense比較、H6のstate-action係数までlocal確認 | GPU経路をH8/H10で確認し、H12の候補$L_D$ごとに$C$を決定 |

- [2026-10-06](2026-10-06.md)：H4追加6距離・最大12 workersの契約確認と、サーバー側local契約案/schema/129合成検査。4判断待ち、科学・port・commit/push未認可でSTOP。


## H4候補間compile投入・12 worker明示再実行

利用者の増員再実行指示により[run03 source・認可・検査記録](../track_a_h4_cross_candidate_run03.md)を追加した。候補内2回路の完了待ちで4 workerがidleとなる問題を、候補間bounded queueで修正。旧run02はworker failure STOPで全証跡を保持し、旧6入力を再生成せず利用する。49人工job/metadata testsはlocal PASSで科学的結果ではない。12 workerのfresh CPU/memory/容量/hash検査後だけ一度起動しMAP_COMPLETE_STOP。旧累積bytes/wall/actual invocationsを引継ぎ、科学条件・compiler・上限は不変。


## H4 run04：identity hash分割・5秒監視維持

[run04固定sourceと再実行binding](../track_a_h4_streaming_monitor_run04.md)を追加した。run03はmonitor STOPで旧全証跡を保持。人工JSONで旧一括encoderのmonitor interval/freshnessを再現し、新streaming encoderは同じdigestで全監視PASS。65 local tests PASSは実装証拠で科学結果ではない。旧6入力を再生成せず、bytes/wall/7 consumed invocationsをcarryし、CPU12件・12 workerのfresh検査後に一度実行してMAP_COMPLETE_STOP。
