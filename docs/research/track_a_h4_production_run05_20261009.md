# H4 run05：carry0とSTOP後の修正・再実行を明示認可

利用者は過去失敗分の予算除外、STOP後もチャットを終了せず原因調査・修正・再実行の継続、および「では本計算を行って」を明示指示した。
今回から新規map attemptごとにcarry0で会計を開始する。旧charge/journal/ledger/STOP/inputを保持し、過去の記録を書き換えない。
各runのexclusive one-shot/own-run fail-closed/cleanupは維持し、STOP後は所有14identityの退出を確認してから新しいrun/source/bindingへ進む。
原因を直さない同一run再開やblind retryは行わず、通常の必要修正と再実行では追加利用者承認を要求しない。H4 signal/compile map完了後は終了、次stage未認可。

SOURCE `9d1471aff5840a76fa1579f4e14e71e62b8497a3`、closure52。branch `track-a-h4-production-run05-20261009`。
旧SOURCE cd162e9305143d81c28908b9732f8eef12cd6b89のlibrary cache v2を引き継ぎ、stevedore disk cacheをown-process内で無効化する。
旧compiler完全同一性と修正後の実科学compile成功は未検証。追加人工compileは利用者方針で省略する。
新gate/schemaはcarry0とuser authority/recovery policyを閉じたfieldsで結合し、run03停止証拠を歴史22件の値のまま検証する。
限定16 pure metadata tests PASS、fail/error/skip0、wall1.069037秒、peak RSS26,214,400B。
回路build/compile/transpile/科学array/実worker/affinity/GPU0。以前の8 cache testsと旧benchmark/synthetic campaignは再実行しない。

新run：`h4-newhost-signal-compile-20261009-run05`。新output/controlはprivate home evidence配下、未使用のままartifact固定する。
次回carry0 actual/bytes/wall、全74784 logical wrappersに対する保証actual最悪74784、静的charge bound8,736,971,632B（8.136939GiB）。
承認済み17GiB/74805 actual、72h、driver/worker各AS/RSS8GiB、headroom16GiB、monitor5秒を保持。
worker12 CPU[2,4,5,6,8,9,10,11,12,13,14,15]、driver16、observer18、own-run限定affinity、数値内部thread1、Qiskit num_processes1、Python -P -B。
observer AS256MiB/RSS64MiB、admission120.25GiB。72h observer reserve/control/journal/temp-final/block/inode/metadata余裕を含む。
今回必要物理容量5GiB/301000 inodes。既存private venvにinstall/upgrade/設定変更なし、共有環境/他job/GPU変更なし。

SOURCE/profile/input streaming bytes/旧hostとrun01/02/03 native STOP proof/利用者認可/plan/authのmetadata binding PASS。
旧13identity proof・既知9SHA・凍結NPZ6/generation freezeの受領済み証拠は保持し、NPZを再生成しない。
SOURCEをseedに再結合し、旧partial/random/cacheの科学結果を混合しない。
独立最終review承認、軽量REVIEW artifact commit固定、直前CPU/memory/pressure/OOM/filesystem/inode/quota、未使用output/control/one-shotの合格後に起動する。
動作中は既定observerが5秒以内の監視/own-run停止を維持し、agentも結果/first STOPを観察して原因修正を継続する。

[固定source・user authority・plan/auth/review・起動argv](../../artifacts/resource_applicability/track_a_h4_production_run05/2026-10-09/README.md)。
[前run03のSTOPとcache修正](track_a_h4_entrypoint_cache_fix_20261009.md)、[carry除外方針](track_a_h4_per_attempt_budget_20261009.md)を履歴に保持。
NPZ/実runtime/checkpoint/cache/credential/内部SSH/private utilityはcommitしない。

## run04 STOPと必要な診断修正

run04はSOURCE d31b51080665a7eea806ff1adc17f3e0d19151fc、artifact cde398580dda6b8d34f999c06a05288128637095で一度起動した。
685.676682秒でmemory_pressure STOP、exit143、12 actual予約/完了0/signal0、14identity残存0を2回確認。
元費用4,263,792,756Bと保守的wall742.2718616949767秒をrun04の履歴に保持し、run05へ加算しない。
停止時memory値は旧observerに未記録なので数値・発生scope・正確な終了時刻を補完しない。
STOP後のhost/shared ancestor PSIが0へ戻り、自ユーザーslice PSI累積0/session1μsだったという観測だけを保持する。
共有ancestor由来を示唆するが停止時の直接証明ではない。他ユーザーのprocess調査・停止・設定変更は行わない。
resourcesにhost/全nonroot scopeのPSI map、observerにavailable/host_available/PSI/scope/OOM値snapshotを追加し、
frame8192Bとfirst STOP保存を限定fixtureで検査。既存のpressure0要求・OOM差分・headroom・8GiB役割上限・5秒gateは変更しない。
再実行前にはPSI0の短い連続観測を保存し、fresh CPU/memory/pressure/OOM/容量/inode/quotaに合格したときだけ新runでexecする。
run04の一時pressureはsource科学errorとは扱わず、共有条件の回復後に継続認可に基づき再実行する。

独立最終review：`PASS_READY_FOR_IMMUTABLE_ARTIFACT_AND_FRESH_RUN05_LAUNCH`、blocking implementation findingsなし。
SHA256 `2434bafdfc21dc1ed9c46edbd16b6a8a5fff1623db7628c547835bc54fd70a30`。SOURCE52/profile/input/歴史native STOP/carry0/認可scopeを別担当で照合。
