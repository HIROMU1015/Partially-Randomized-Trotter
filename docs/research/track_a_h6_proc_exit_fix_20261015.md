# H6 pilot：/proc退出競合の修正とrun04-v3再実行

2026-10-15。利用者の再実行・未完了時の必要修正指示に従う。SOURCE `2e733d8a683d415792ce92a56c7d149b78f42716`、branch `track-a-h6-pilot-proc-exit-fix-20261015`。サブエージェント0、短い同担当binding確認であり独立reviewとは扱わない。

[修正・92検査・実退出確認・新grant・起動証跡の入口](../../artifacts/resource_applicability/track_a_h6_proc_exit_fix_v1/2026-10-15/README.md)。[前回の独立観測修正](track_a_h6_external_monitor_fix_20261015.md)、[run03停止証跡](../../artifacts/resource_applicability/track_a_h6_external_monitor_fix_v1/2026-10-15/run03_stop_result_v1.json)。

## 停止事実と原因

run03はSOURCE b73580dd…で全3構成のnative correctnessと12 probesが合格。最大discrepancyはp15q8の8.94286541500109e-14、既存主gate約1.1e-9以内。新しい独立観測は全期間継続し、最大開始間隔約1.024秒、最大所要約.012秒、observerのfirst STOPなしだった。

6 compile childrenが全てactual callへ入った後、親の/proc status読取が`KeyError: VmRSS`となり停止。最初のp5q1 cosineはnative exit0/PASS payload、RZ10670・CX4764・depth4184、残る5 workerは親cleanupのOWN_SIGNAL15でexit2。これは部分worker payloadであり、受理済みpilot結果ではない。全6 reaped/errors[]、driver wait2/recorder0/元identity不存在、launcher wall1109.924569182098秒を固定した。

正常退出と同時にmemory fieldが欠けた観測は、退出時にmemory mapping情報が消える競合と整合する。kernel内の正確な関数・時系列のtraceは未取得で、exit_mmが今回の位置だったと証明したものではない。Stateやwait4(WNOHANG)の更新前にmemory fieldが欠ける場合を含め、混在した/proc snapshotを単なるKeyErrorにしないよう修正した。生存中の欠測を0や成功で補完してはならない。

## 必要修正

親backendはVmRSS/VmSizeだけが欠けた場合にwait4を再確認する。reapedならnative statusとpipe payloadを通常の経路で処理する。未退出ならownerを再確認し、最後の正常資源観測から最大5秒だけpendingとし、他workerの観測・drain/reapも継続。5秒超の生存欠測はCOMPILE_MEMORY_OBSERVATION_MISSINGでSTOP。正常観測時だけfreshnessを更新する。非memory field、他IO/permission error、所有不一致を成功扱いしない。ESRCHだけを退出候補とし他errnoは伝播する。

既存metadata launcherの独立observerもVmHWM/VmSizeの一時欠落をfresh leaseとして返さず、最後の正常観測から5秒超ならSTOP。既存のglobal観測間隔5秒・resource上限・child parent-deathとown cleanupを維持。科学provider、blueprint、input、compilerとshot/u数式は変えていない。

新8＋既存84＝92限定tests PASS、wall約1.41秒、RSS32MiB、single process/thread1/science imports0/実compile0。mockは正常退出reap、live pending→5秒STOP、owner変更、非memory欠測、zombie/proc消滅、permission/IO errorを確認。さらに最小2child/全体3process/thread1、AS256MiB/RSS128MiB/CPU30秒/wall20秒/output1MiBの事前計画で、native exit0/wait4/reap/pipe EOFとFD閉鎖・元identity不存在を確認。wall約.21秒、compile0、科学import0。6実worker stressや新人工compileは行わない。

source2 runtime files＋新2tests/fixture＋harness変更をSOURCE commitへ固定し、closure13を再結合。旧run01/run02/run03・診断のknown/unknown、run03 actual compile entered6/PASS payload1/accepted0をcarryへ保存。結果再利用・返却/reset0。

## 新実行と現在状態

固定候補はH6 linear1Å/STO-3G/既存tol-only DF19/T=.8/S2、prefix/q 5/1・10/2・15/8×cosine/sineの6 compile。scientific provider87b6b9f…、profile34273files、入力5件、compiler basis rz/sx/x/cx/opt1/seed17、AS32GiB/RSS9GiB/4h stage wall・aggregate CPU、admission240.5GiB等を維持。新SCF/DF/state/signal指数作用/trajectory生成0。matched B0/全面map/次stage未認可。

run04-v1/v2は直前CPU gateで混雑を検出し未起動。元grantはCANCELLED_BEFORE_LAUNCH markerで廃止し、科学操作0のまま保存した。CPU選定を5回×1秒、全区間最大使用率5%以下の物理coreへ厳しく変更し、新bindingを作成。直前15%以下のgateを緩和しない。origin/共有設定・venv・他job変更0。

run04-v3は全SOURCE/profile/input/carry・memory/PSI/OOM/FS/inode/quota・未使用output/control/one-shotを再照合して起動。CPU driver62、workers61/60/59/56/49/47、内部thread1、metadata launcherは既存role。run ID `h6-native-b0i-pilot6-20261015-run04-v3`、launcher PID1470933。起動証跡はbundleの`run04_launcher_identity_v1.json`、実ログはhome配下のprivate evidence `h6-native-b0i-pilot6-run04-v3/control04/`、結果は同`run04/`。

本資料は起動・修正の記録であり、本計算完了や6軸成功の主張ではない。全6軸完了またはfail-closed STOPで終了し、結果/native status/own cleanupを監査する。runner内部のautomatic_retry=falseと次stage未認可を維持し、必要修正と新grantの根拠は今回の明示指示に保持する。
