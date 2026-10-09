# H4 run07：利用者承認のhost-only pressure条件で再実行

利用者は提示した修正案に「この条件を承認して再実行」と明示回答した。元D4のPSI非zero即STOPに対し、
host-only full.avg10<1.0%、全visible nonroot cgroup PSI0、OOM基準一致、effective available120.25GiB以上、fresh<=5秒を満たす場合だけ継続する変更を承認した。
全cgroupのどれかが非zero、host>=1.0%、OOM増分、namespace/hierarchy/scopes欠測・不整合、余裕不足はSTOPする。
host PSI0の通常headroom16GiB、全role AS/RSS8GiB、observer AS256MiB/RSS64MiB、interval/duration/staleness5秒は維持する。
1%が最適であることや他jobが原因であること、本体完走は未証明。承認された条件を独立reviewとfixed sourceで実装し、診断値を保存する。
元contract37files/科学fingerprintは保持し、別pressure amendmentのprofile/authorityをplan/auth/reviewにbyte SHA結合する。

SOURCE `1dbdd1be2133a59ab81c7b81a6074af0880f72a3`、closure62。branch `track-a-h4-production-run07-20261010`。
proposalの純粋predicateを閉じたruntime profileで使い、fresh/OwnedRun二次gate/observer子/driver monitorが同じ条件を適用する。
actual resources.observe_memoryから完全なinitial-v2 namespace/全祖先scopesを観測し、原driverのOOMとordered hierarchyを固定baselineとして子へ渡す。
observer起動直前・起動時のOOM増加やscope変更を新baselineで隠さずSTOP。profileなしlegacy経路はstrict PSI0のまま。
科学式・回路構築・compiler options・ledger/parallel順序・library cache v2・入力は変更しない。source固定に伴うseed再結合だけ実施。

限定33pure tests PASS、fail/error/skip0、wall0.383531秒、peak RSS29,884,416B。
通常1process/thread1、AS2GiB/RSS128MiB/wall30s/output1MiB。observer/worker/thread/affinityはmock、科学array/回路build/compile/transpile/実child/GPU/本体0。
最初の2errorと次の1errorを保存し、mockroot型とrun06 exit143定数の誤変換を修正後に全合格。旧benchmark128/synthetic28/64の再実行なし。
追加人工compileは利用者指示で省略。新pressure下の実科学compile成功はこの人工合格から推定しない。

run06はhost PSI0.18/nonroot0/OOM増分0/available約445GiBで約856.585秒にSTOP。4予約/完了0/signal0、6owned identityを2回ABSENT確認。
原receipt8476Bと全10filesのbytes/SHA・ledger chain・journal charge4,263,786,622B・one-shotを照合し、元の値を保持する。
正確な終了時刻・pressure発生者は補完しない。run01〜06/旧hostのproofを検証するが、失敗費用は明示指示に従い次attemptへ加算しない。
新carry0 actual/bytes/wall。新run `h4-newhost-signal-compile-20261010-run07`、未使用output/control/one-shotへ一度起動する。
4workers CPU[2,4,5,6]、driver16、observer18、own-run限定affinity、数値内部thread1/Qiskit num_processes1/Python -P -B。
承認済み17GiB/74805 actual/72h、全map保証actual74784・静的charge bound8,736,971,632B、物理容量5GiB/301000 inodesを維持。
既存venvにinstall/upgrade/設定変更なし、共有環境・cgroup/sysctl/他job/GPU変更なし。新work/log/tempはhome内だけ。
旧partial/cache/random結果を混合しない。凍結6NPZ/generation freezeは再生成せず、科学arrayは認可された本計算内だけで読む。

独立最終review・SOURCE62/profile/input/pressure authority/carryの一致・immutable artifact固定後、直前のallowed-pressure安定観測と
fresh CPU/memory/PSI/OOM/filesystem/inode/quota・未使用output/control/one-shotを再確認して実行する。
STOP後の通常調査修正とfresh再実行は継続認可、同run再開/盲目的retry/次stageは不可。完了時はMAP_COMPLETE_STOP。
利用者は処理開始後にチャットをいったん終了してよいと明示している。終了後は既定observerが監視・own-run停止を担う。
[固定source・承認・plan/auth/review・起動argv](../../artifacts/resource_applicability/track_a_h4_production_run07/2026-10-10/README.md)。
[旧未承認proposalと元契約](track_a_h4_host_pressure_policy_fix_20261010.md)を履歴として保持する。
NPZ/実runtime/checkpoint/cache/credentials/内部SSH/private utilityはcommitしない。

独立最終review：`PASS_READY_FOR_IMMUTABLE_ARTIFACT_AND_FRESH_RUN07_LAUNCH`、blocking implementation findingsなし。
SHA256 `c57d1333eabbf99e0450495aed64ca1069496393ff91140e44346e408c8cc060`。SOURCE62/profile/input/歴史native STOP/carry0/認可scopeを別担当で照合。
