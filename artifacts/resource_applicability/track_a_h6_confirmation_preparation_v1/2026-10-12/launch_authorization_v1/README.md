# H6独立32確認：認可済み4worker run実行中

run ID `h6-independent32-confirmation-v1`、coordinator PID20241、SOURCE `73018cbcc39803890974c7ee51401063e6c8daf6`。
[起動・引継ぎ契約](../../../../../docs/research/track_a_h6_confirmation_launch_v1.md)。

- [ユーザー認可根拠と未改変実行command](user_authorization_receipt_v1.json)
- [承認済みexact grant](authorization_approved_v1.json)
- [SOURCE/profile/input/carryとfresh資源gate](prelaunch_gate_v1.json)
- [初期native起動観測](launch_observation_v1.json)
- [将来の完了group限定引継ぎ方針・未実装事項](future_handoff_intent_v1.json)

固定8構成×独立32＝256 group/512 wrapper、4worker・thread1・wall capなし、累積trajectory840承認。他上限/carry/科学条件/compiler不変。完了/STOP後終了、自動retry/追加sample/次stageなし。
ハッシュ再利用・8workerはまだ未実装/未起動。将来の切替は完了whole groupだけをsource別に検証して再利用し、partialは再計算する。切替の追加budgetと新資源はその時点で照合する。実確認の完了や科学的成功はまだ主張しない。

read-only status： `/home/AbeHiromu/projects/h6-handoff-evidence/20261012/confirmation-launch-v1/MONITOR_STATUS.py`
coordinator log： `/home/AbeHiromu/projects/h6-handoff-evidence/20261012/confirmation-launch-v1/coordinator.log`
raw output： `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-confirmation-preparation-20261012/artifacts/resource_applicability/track_a_h6_independent_confirmation_v1/2026-10-12/launch_confirmation_v1/`
raw runtime/cache/NPZをcommitしない。Git公開は認証待ち、remote SHA=null。
