# 未承認の実行・STOP・cleanup契約

## 入口と承認前の拒否

新runnerは `h4-b0i-pilot6-authorization-v1` と固定plan/source/digestに結合した新authorizationだけを受け付ける。
planとauthのapproved/runtime_authorizationがともにtrue、明示CPU集合と絶対commandが一致し、
approval evidence SHAと別reviewのplan/auth digestが揃わなければ拒否する。
既存H4/H8 grant、旧observer role承認、旧cacheを流用しない。
現在のplan/auth/reviewは未承認であり、拒否はsource/profile検査、入力アクセス、lock、mkdir、import、spawnより前。

## 承認後に残る操作（code開発ではなく実行binding）

1. GPTの採否・pilot必要性判断と、利用者の新しい6件実測・資源・CPU・launchの明示承認。
2. 固定SOURCEを変えず、approved plan/authへ承認evidence・3 role CPUs・絶対commandを結合し、digestと最終reviewを更新。
3. source48 closure/profile45 RECORD/compiler entrypoint/options、保存metadata、対象3入力file bytes/SHAを再確認。
4. fresh resource/CPU/block/inode/quotaに合格し、未使用output/controlとO_EXCL lockが成立したときだけone-shotを開始。
5. 承認後の小member decodeと回路build、6軸別compile、native receiptを保存し、完了またはfail-closed STOP後に停止。

scientific inputの再生成・新signal・sampling、scope追加、次stageは含まない。
6件のあと10/240件へ進む条件分岐・schedulerを実装していない。
現在の `absolute_launch_command=null` を、仮の実行commandで補完しない。

## Fresh admission

- 3 roleのCPUは利用者が明示した異なる3 physical cores。online・scheduler/process cpusetの共通部分に含まれ、受動sample busy<=20%。空きを使用権とみなさない。
- effective memory>=120.25GiB。driver8＋worker8＋headroom16＋observer AS.25GiBより保守的な旧120GiB floorを採ったproposal。
- host full PSI avg10<1%、全non-root full PSI=0。v1 pressure policyをこの新pilotの未承認proposalとして使用する。H8や既存runのv2 grace方針を変更しない。
- cgroup hierarchy/namespace/全OOM counterを保存し、増加・scope変更・欠測でSTOP。host `/proc/vmstat:oom_kill` もfresh admission前から保持し増加でSTOP。旧counterを0へresetしない。
- native quota GETの全scopeがKNOWN、DISABLEDまたは十分なACTIVE残量。unknown/permission errorをdisabledへ置換しない。
- stage filesystemに320MiB・256 inodes以上。出力256MiBと終端free-space64MiBを別に確保する。
- 観測age/duration<=5秒。old snapshotやこの資料作成時のCPU sampleをlaunchへ再利用しない。

今回のsnapshotは使用許可ではない。launch直前に再観測し、不合格なら起動しない。

## 実行中

driver1、worker1、独立observer1。内部thread1、Qiskit num_processes1、Python -P -B。
affinity変更は新auth内の各自PIDだけ。PID/starttime/UID/parentとstable pidfdでown-runを確認する。
observerはperiod1秒、interval/duration/staleness<=5秒、driver phaseと最初の停止理由を別保存し、重いbuild/hash/compile中も独立して継続する。
driverのpipe I/Oはnonblocking/selectとtask600秒/stage3600秒のdeadlineで制限する。
stage SIGALRMはdriver自身だけ。task/frame不完全・worker退出・monitor failure・RSS/AS超過・PSI/OOM・容量/inode不足・quota不明はfail-closed。
driver I/O loopでもhost OOMとoutput/control/lockのallocated block/inode合計を観測する。

入力の全NPZ byte SHAを確認した後、workerは回路loweringに必要なone/G/lambdas/nuclearの小memberだけを読む。
kept blocksの既存diagonalize/basis builderを使い、tail eigensystem、H/state arrays、原source `_prepare` のdense reconstructionやsignal solverを呼ばない。
DF/stateを再生成する操作ではなく、凍結係数から既存B0と同じ回路をloweringする段階である。
tail scalarは先行診断の固定値を使用し、lower conventionのtrace算術との整合を確認する。

## 予約・complete-only

taskごとに永続reservationをfsyncしてからworkerへdispatchする。未知task、duplicate、7件目は拒否。
失敗・未解決reservationは返却しない。0 scientific cache reuseで、actual最悪追加6。
旧B0 RZは差を説明するreferenceだけ。新metricsは各axisの新full wrapperから取得する。
numeric FPとsource/input/phase/control/template/axisのvalidation FPを別保存し、compilerによるinput mutationを検出する。
COMPLETE/task ID/source/plan/no-old-cache/metric schemaに一致しない結果を受理しない。
worker errorはbounded frameに元exception/phaseを保存し、parent STOPまでdispatchせず待つ。
後続のEOF/schema errorで元原因を隠さない。

taskのCOMPLETEだけではrun最終受理にならない。6件の受理とnative終了・reap・FD/pipe終了、
`PILOT6_COMPLETE_STOP` のterminal receiptを合わせて受理する。部分runは保存してSTOP、retry/resume/cache reuseは自動実行しない。

## 正常終端と異常cleanup

既存observerはworkerの予期しない退出をSTOP扱いする。その退出/reap競合を避けるため、新runnerは
**6件を受理してworkerがidleのまま、fresh ping→owned observerの意図的SIGTERM→native waitを先に行う**。
正常observer exit codeは、この新契約では意図的終了を示す `-SIGTERM`（通常-15）。終了理由・identity・6 task IDを別receiptに保存する。
その後は新computeをdispatchせず、tiny shutdown frame→worker exit0→wait/reap→FD/pipe closeだけを行う。
異なるobserver exit、途中の退出、退出/reap未確認を正常と扱わない。正常終端もコードから予測したcodeを保存せず、将来のnative wait結果を保存する。

失敗時はfirst STOPを保持してown childのpidfd TERM、bounded wait、必要なら同じ所有pidfd KILL、wait/reap、pipe/FD終了を行う。
ESRCHは既に退出として後続cleanupを続ける。非ESRCHは成功扱いせず、wait/FD終了を続けてcleanup不合格を記録する。
bootstrapでpidfd登録が成立する前の直接Popen childだけは、その未reap childのPopen所有関係で終了・waitする。
通常の所有不一致に対するforeign PID fallbackはない。
observer close、二重cleanup、partial IO、元STOP reason保持は人工mockで検査した。

one-shot lockはO_EXCL/O_NOFOLLOW、namespace/plan digestを保存し、終了後もunlinkしない。
正常/異常後ともnext_stage=false、retry=false、mandatory_stop=true。
今回native live-process競合・実H4 compiler結果の測定は行っておらず、合格範囲は人工/mockと静的実装である。
