# H4 run06：carry0とSTOP後の修正・再実行を明示認可

利用者は過去失敗分の予算除外、STOP後もチャットを終了せず原因調査・修正・再実行の継続、および「では本計算を行って」を明示指示した。
今回から新規map attemptごとにcarry0で会計を開始する。旧charge/journal/ledger/STOP/inputを保持し、過去の記録を書き換えない。
各runのexclusive one-shot/own-run fail-closed/cleanupは維持し、STOP後は各attemptに登録された全owned identity（旧run05は14、新run06は6）の退出を確認してから新しいrun/source/bindingへ進む。
原因を直さない同一run再開やblind retryは行わず、通常の必要修正と再実行では追加利用者承認を要求しない。H4 signal/compile map完了後は終了、次stage未認可。

SOURCE `697843fbbd2a2aa7224261aa26f8da141ca57687`、closure55。branch `track-a-h4-production-run06-20261010`。
旧SOURCE cd162e9305143d81c28908b9732f8eef12cd6b89のlibrary cache v2を引き継ぎ、stevedore disk cacheをown-process内で無効化する。
旧compiler完全同一性と修正後の実科学compile成功は未検証。追加人工compileは利用者方針で省略する。
新gate/schemaはcarry0とuser authority/recovery policyを閉じたfieldsで結合し、run03停止証拠を歴史22件の値のまま検証する。
限定21 pure metadata tests PASS、fail/error/skip0、wall1.193839秒、peak RSS29,360,128B。
回路build/compile/transpile/科学array/実worker/affinity/GPU0。以前の8 cache testsと旧benchmark/synthetic campaignは再実行しない。

新run：`h4-newhost-signal-compile-20261010-run06`。新output/controlはprivate home evidence配下、未使用のままartifact固定する。
次回carry0 actual/bytes/wall、全74784 logical wrappersに対する保証actual最悪74784、静的charge bound8,736,971,632B（8.136939GiB）。
承認済み17GiB/74805 actual、72h、driver/worker各AS/RSS8GiB、headroom16GiB、monitor5秒を保持。
worker4 CPU[2,4,5,6]、driver16、observer18、own-run限定affinity、数値内部thread1、Qiskit num_processes1、Python -P -B。
observer AS256MiB/RSS64MiB、admission120.25GiB。72h observer reserve/control/journal/temp-final/block/inode/metadata余裕を含む。
今回必要物理容量5GiB/301000 inodes。既存private venvにinstall/upgrade/設定変更なし、共有環境/他job/GPU変更なし。

SOURCE/profile/input streaming bytes/旧hostとrun01/02/03/04/05 native STOP proof/利用者認可/plan/authのmetadata binding PASS。
旧13identity proof・既知9SHA・凍結NPZ6/generation freezeの受領済み証拠は保持し、NPZを再生成しない。
SOURCEをseedに再結合し、旧partial/random/cacheの科学結果を混合しない。
独立最終review承認、軽量REVIEW artifact commit固定、直前CPU/memory/pressure/OOM/filesystem/inode/quota、未使用output/control/one-shotの合格後に起動する。
動作中は既定observerが5秒以内の監視/own-run停止を維持し、agentも結果/first STOPを観察して原因修正を継続する。

[固定source・user authority・plan/auth/review・起動argv](../../artifacts/resource_applicability/track_a_h4_production_run06/2026-10-10/README.md)。
[前run03のSTOPとcache修正](track_a_h4_entrypoint_cache_fix_20261009.md)、[carry除外方針](track_a_h4_per_attempt_budget_20261009.md)を履歴に保持。
NPZ/実runtime/checkpoint/cache/credential/内部SSH/private utilityはcommitしない。

## run05停止と自分の負荷を下げる再実行

run05 SOURCE9d1471aff5840a76fa1579f4e14e71e62b8497a3/artifact34c3fd46cd5dede93473f14187b3e92471f315eaで1105.977738秒の観測時にmemory_pressure STOP。
新diagnosticでhost PSI full avg10=0.18%、全nonroot cgroup=0%、available428727332864B、OOM増分0を保存した。
最大worker AS7.651GiB/RSS7.248GiBは8GiB以内、completed0/signal0/actual reservation12、元charge4263792756B。
全14identityの退出を2回確認、exit143。正確な終了時刻やpressure発生者は断定/補完しない。
worker12で2回pressure STOPが出たため、許可済み最大12の内側で自分の同時compile数を4に減らす。
CPU[2,4,5,6]は元許可集合の部分集合、driver16/observer18、各role AS/RSS8GiB等のcapは不変。
減数による資源消費上限を下げ、memory admission floor120.25GiBと全host/nonroot PSI非zero停止を維持する。
科学条件/回路/乱数設計/compiler options/profileを変更しない。SOURCE固定に伴うseed再結合だけ行い、旧partial/cache結果を混合しない。
共有環境・他ユーザーのjob/affinity/設定は変更しない。global pressureを隠したり無視したりしない。
21pure tests PASS、限定実worker stress/追加人工compileなし。次の新run06はcarry0、元run05費用/証拠は履歴に保存。

独立reviewの二次起動admission指摘を安全側に修正し、runtimeの両admission gateを120.25GiBに固定。4worker64GiBをobserver/pool起動前に拒否する統合mock回帰PASS。SOURCE再固定後21pure tests PASS、実child/追加compile0。前SOURCEと20PASSの証拠も保持。

独立最終review：`PASS_READY_FOR_IMMUTABLE_ARTIFACT_AND_FRESH_RUN06_LAUNCH`、blocking implementation findingsなし。
SHA256 `2202a3d7b37422acc9b4ad1a57a0c83e26fc112c94103b8389bdde1f5be489e7`。SOURCE55/profile/input/歴史native STOP/carry0/認可scopeを別担当で照合。
