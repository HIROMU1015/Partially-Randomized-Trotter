# H4新host A案：監視修正・人工検証・source固定

H4_NEW_HOST_MONITOR_FIX_A_SOURCE_FROZEN_PRODUCTION_UNAUTHORIZED_STOP。既存private venvを変更せず準備用だけに採用した。32人工tests PASS、fail/error/skip0。本計算未認可・未開始。

branch `track-a-h4-new-server-monitor-fix-a-20261007-02`、起点 `814212cb1c25f27e3300294e72fef452d953fbc9`、最終SOURCE `b2a5ad89e8b39d72716f7ddb17d263bd0cdedb45`。SOURCE/test commitと別REVIEW_BUNDLEを分離する。
REVIEW_BUNDLEのactual SHAとremote SHAは固定後の外部公開receipt・最終報告へ記録する（self-referential SHAを草案へ捏造しない）。
資料入口は [bundle](../../artifacts/resource_applicability/track_a_h4_new_server_monitor_fix_a/2026-10-07/README.md)、[actual25 source closure](../../artifacts/resource_applicability/track_a_h4_new_server_monitor_fix_a/2026-10-07/source_freeze_v1.json)、
[人工結果](../../artifacts/resource_applicability/track_a_h4_new_server_monitor_fix_a/2026-10-07/test_results_v1.json)、[未seal binding](../../artifacts/resource_applicability/track_a_h4_new_server_monitor_fix_a/2026-10-07/binding_draft_v1.json)。
worktree `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-monitor-fix-a-20261007-02`、外部log/evidence `/home/AbeHiromu/projects/h4-handoff-evidence/20261007/monitor-fix-a`。既存worktreeを保持し、全新fileはhome内。/tmpを使わず共有設定を変更しない。

## 修正と互換性

旧source19の変更はcircuits/identity/executionの3件、16件はblob/bytes不変。新streaming/observer moduleと専用runner/auditor/tests/oracleを含め25件のclosureをactual SOURCEのblob/hashで固定。
ndarray.tolistと全matrix exact tree、全instruction listの先行生成を避ける。行優先のarray scalarとinstruction/definitionを遅延走査し、encoding buffer上限64KiB、matrix metadata memoはcall-local最大64件、encoded cache0。
dtype/shape、phase、control state、condition、ordering、signed zeroを保持し、nonfinite/symbolicはstreaming消費境界でSTOP。
`serialize()`は遅延recordとなるため、旧eager APIの早期error時点は変更した。parameter/circuitは一回の走査中readonlyであることが必要。科学builder/metricsは変更しない。

独立stdlib observerはdriverのGIL/GCから分離したprocessでresource observationを続ける。private bounded JSON socket、UID/PID/starttime/親子/pidfd、全登録workerのRSS/AS、host/cgroup memory/pressure/OOMを確認。
interval・観測所要時間・staleness・driver phase/sequence/monotonic・最初のfailureを別fieldで保存する。5秒超過はfail-closed。
first failureをfsyncしてからown-runへsignalを送る。driver退出後は元driverのpidfdが退出を証明した場合に限り親PID変更を許容し、登録workerのUID/PID/starttime/pidfdを再照合して停止する。driver側はobserverのdurable原因を優先し、observer自身の死/EOFは別bounded first-stop fileに保存する。shutdown/reap/FD closeは人工caseで確認。
本番経路は新observer roleの別approved/runtime_authorizationを要求する。既存run05の環境/CPU/source/auth gateは解除せず、本計算bindingへ準備profileを採用していない。

## 人工検証

256×256の人工complex parameterを含む8-system+ancilla相当の9-qubit人工回路で、cosine/sine旧/new canonical bytesとdigestが一致。
bytesは4,458,518 / 4,458,516、digestsは `227aa2434a2b25840dad01d7f4fcf0727f30ba2a076e9ecbd63a8b47d74c20d8` /
`fc6ab3cc424494ea91e2abeceabef451ef25557cff7788f267430ffc547c1c18`。
matrixは人工parameterで、物理operatorの検証ではない。
GIL占有、bounded heavy cyclic GC、serialization中の独立観測を確認。観測I/O5.1秒delayはduration STOP、最初の原因をchild/driverで一致保存。
12 workers所有・RSS/AS、退出/reuse/foreign owner、欠測、pressure/OOM、frame/EOF、output上限はfixture/mock。実workers0。

通常1 test process・内部thread1、実動observer caseのみdriver/observer各1。-P -B、全process限定thread1、Qiskit num_processes1。driver AS2GiB/RSS候補512MiB、observer AS256MiB/RSS64MiB、attempt180秒/16MiB、合計600秒の計画を実行前固定。
最終32 testsのwall 13.889663秒、driver peak RSS 156.57MiB。
observer21観測、max RSS19MiB/AS28.254MiB、max interval0.107148秒。
synthetic3 observerのCPU user+system計0.303875秒。trace/logのraw証跡は外部manifestでhash固定しGitへ入れない。
初回はQiskit依存dillのtempdir探索をguardがopen前に拒否した。以後、driver process内のtempdir問い合わせだけをhomeへ向け、/tmp probeを行わない。venv/共有設定変更0。
追加transpile0、旧synthetic28/64・旧benchmark128件保持、再実行0。人工PASSはproduction成功・旧compiler完全同一性ではない。

## 環境・入力・予算

準備用Python `/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python`、環境profile `47f1525960ac9d5bad9f0dbf3246d4a58b10b5f236deacd1c042990c0d8bd25d`、compiler profile `b45521fd39220079a4475882a5dac7ca37391786d39f74aa8e843e6e75a3a1b6`。
18 version差を保存。旧監査45 RECORD差も保持し、old metadata.read_textの改行正規化hash対new raw bytesという方法を明記した。
同じread_text方式で比較すると22差/23同hashであり、45件をpackage内容が全て異なる証明とは扱わない。旧raw RECORD bytesは未受領。11 installed source hashは一致し、142 distributionsのmetadata/RECORD normalized hashはtests前後不変。
compiler15 optionsは不変、28 defaultsと61 plugin metadataを取得したが、旧環境でのcompiler output equivalenceは未検証。

6 NPZ/freeze/runtime/controlは未受領。state資料だけ受領、実byte streaming SHA照合0、入力再生成0、実NPZ使用0。
carry20 actual invocations /165214360 bytes /5466.188392877579秒、残74764を保持。準備test費用は別明示し、science journalを作成・budget resetしていない。
新SOURCE/environment/compilerで将来seedを再結合する。旧random/partial/cache結果とは混合しない。

observer費用案は [資源・容量草案](../../artifacts/resource_applicability/track_a_h4_new_server_monitor_fix_a/2026-10-07/observer_budget_proposal_v1.json)。候補AS256MiB/RSS64MiB、12-worker admission120.25GiBは未承認。
72h/1秒周期・最大8KiB/recordでtrace上限2123399168 bytes、driver first-stop含むreservation 2123407360 bytes、保守的charge 4246814848 bytes（約3.955GiB）。
旧stage charge見積りとcarryを単純加算した条件付き見積りは8984725656 bytes（約8.368GiB）で10GiB内。
旧JSONサイズはhard gateではないので容量PASSにはしない。物理容量候補5.5625GiB/560008 inodes、input/control転送容量・quota・全campaignは別検証が必要。
driver/worker AS/RSS各8GiB、headroom16GiB、monitor5秒、10GiB/72h/74784上限は不変。

## STOPと残るreview

allowed_cpus=[]、approved=false、runtime_authorization=false、sealed=false、absolute_launch_command=null。
本計算environment/compiler、入力/output、observer追加role/CPU配置、容量・quota、final review、明示launchは未認可。
実12-worker stress、全74784/72h、driver実退出・実worker群停止、power-loss durability、kernel I/O停止とdriver GIL停止が同時に起きた場合の5秒以内停止は未検証。
旧compiler equivalenceに必要なpaired fixture・件数・予算は未固定、今回追加予定0。残synthetic36件を自動消費しない。
未承認の新roleを受け付けるproduction auth/schema/gate/profile bindingは次reviewで固定する。今回はこのsource/人工結果・未承認草案を公開してSTOP。
