# H4一度のsignal/compile map：利用者承認と固定実行artifact（2026-10-09）

**利用者が候補環境・CPU/observer・累積actual +20と一度のmap起動を明示承認した。最終artifact固定後、直前fresh条件PASSなら追加確認なしで一度起動する。**
起点branch `track-a-h4-native-proof-seal-20261009` のorigin SHAはREVIEW `5de745199748a6523ba84b07a0ea61b2b831815b` に一致。
SOURCE `6bd1ba01cd71ec3e2071082963c9f07478dada9a` を変更せず、独立worktree/branch `track-a-h4-authorized-launch-20261009` に認可artifactを固定した。
既存proof受領・13identity解析・回帰・benchmarkを再実行していない。新準備campaign/追加transpile0。

本書を一つの入口とする。
[明示利用者認可](../../artifacts/resource_applicability/track_a_h4_authorized_launch/2026-10-09/USER_AUTHORIZATION_v1.md)、
[plan](../../artifacts/resource_applicability/track_a_h4_authorized_launch/2026-10-09/plan_authorized_v8.json)、
[auth](../../artifacts/resource_applicability/track_a_h4_authorized_launch/2026-10-09/authorization_v8.json)、
[review](../../artifacts/resource_applicability/track_a_h4_authorized_launch/2026-10-09/review_v8.json)、
[独立最終review](../../artifacts/resource_applicability/track_a_h4_authorized_launch/2026-10-09/independent_execution_review_v1.json)、
[binding](../../artifacts/resource_applicability/track_a_h4_authorized_launch/2026-10-09/authorization_binding_v8.json)、
[実行手順](../../artifacts/resource_applicability/track_a_h4_authorized_launch/2026-10-09/EXECUTION_PLAN_v1.md)、
[commit対象](../../artifacts/resource_applicability/track_a_h4_authorized_launch/2026-10-09/commit_inventory_v6.json)を参照。
[前technical seal](track_a_h4_native_proof_seal_20261009.md)は実行未認可だった時点のsnapshotとして保持する。

## 認可されたscopeとbinding

既存private venv/profile/compilerを変更せず本番採用。旧compiler output完全同一性は未検証のまま記録する。
worker12 CPU `[2,4,5,6,8,9,10,11,12,13,14,15]`、driver16、observer18、own-run限定affinity、数値library thread1/Qiskit num_processes1/Python -P -B。
observer AS256MiB/RSS64MiB、admission120.25GiB、driver/worker AS/RSS各8GiB、headroom16GiB、monitor5秒。
累積actual capだけ74784→74804へ変更。carry20 actual/165214360 bytes/5466.188392877579秒を保持し、残新actual枠は74784。
10GiB/72h等の他caps、科学条件/compiler options、input/output/control/run IDは不変。失敗予約返却・reset・旧partial/cache再利用なし。
H4 linear neutral singlet/STO-3G/DF12、8system＋ancilla1、T0.8、二次DF-prefix PF/canonical finite-RTE、6距離・218 templates/点・32 paired trajectories・1308 signal/74784 logical mapを一度処理する。
完了またはfail-closed STOP後に終了し、自動retry・worker replacement・入力再生成・次stage・GPU query/use・共有環境/venv/他job変更なし。

SOURCE33 closureのactual blob/checkout、SOURCE/profile/input/carryを固定し、source_rootのみ新worktreeへ更新した。
plan fingerprint `f131fddc1756899543a0edbdcf5be5acfc6a285a32918e32118ffe9a7947e5c7`、auth digest `03ccf63d291e37a64c132abe1f691dd5cf430f5a7396cb4c2d6d80b5bfb29148`。
sealed=true、approved=true、runtime_authorization=true、environment accepted/observer/+20 approved=true、allowed_cpusは14 role CPUのexact集合。
独立reviewの実PASS reportとartifact commitを確認してから起動する。最終actual commitとruntime状態は外部receipt・起動/停止追記で区別して固定する。
現在のnative stop条件PASSと、未記録の過去exit code/正確exit-reap時刻/原boot ID=nullという限界はそのまま維持する。

## 一度の起動と継続監視

run IDは `h4-newhost-signal-compile-20261007-run01`。これはSOURCEの固定run namespaceであり、実開始日時は別のruntime記録へ保存する。
正確なPython/argv/environmentは[runner argv](../../artifacts/resource_applicability/track_a_h4_authorized_launch/2026-10-09/runner_argv_v8.json)。固定runnerを実行ツールの管理sessionで直接execし、追加wrapper/monkeypatch/runtime supervisorは作らない。
immutable artifact固定後に3秒CPU passive sample付きfresh memory/PSI/OOM/filesystem/inode/quota gate、SOURCE/profile/input/carry、unused output/control/one-shotを検査する。不合格なら科学起動0で原因を保存し停止する。
合格後だけ外部submission markerをexclusive取得して一度exec。SOURCE自身のfresh gate・one-shot O_EXCL・output/trace/control費用reserve・own-affinity/監視も維持する。

durable科学logは `/home/AbeHiromu/projects/h4-handoff-evidence/20261007/prelaunch-preparation/control/h4-newhost-signal-compile-20261007-run01/runner.log`。
監視は `/home/AbeHiromu/projects/h4-handoff-evidence/20261007/prelaunch-preparation/h4-newhost-signal-compile-20261007-run01/observer.jsonl`。
launch前stdout/stderrはtool sessionで回収し、保存diagnosticは合計64KiB以内のhome metadataに限定する。科学log8MiB/observer trace/metadataの予約はSOURCE契約どおり。
起動後はone-shot.jsonのdriver PIDとown UID/starttime、observer/registered worker PID・CPU・SOURCE・監視記録を確認し報告する。これをmap成功・全candidate完了と読み替えない。
正常/通常例外のcleanupは停止/wait/reap/FD/pipe終了。observer強制STOPではown identity限定signalでdriverを終了し、driver finallyが走らない場合があるため、得られた実exit statusとown残存をread-only確認し、未記録reapを補完しない。
raw runtime/NPZ/checkpoint/cache/logはGitへ入れない。実行中でもSOURCE/認可artifactを変更せず、軽量な起動/停止観測だけを別記録にする。

起動結果は最終報告・外部runtime receiptへ保存する。起動確認後もSOURCEの既定監視下で完了またはSTOPまで継続し、自動retryせず終了する。
