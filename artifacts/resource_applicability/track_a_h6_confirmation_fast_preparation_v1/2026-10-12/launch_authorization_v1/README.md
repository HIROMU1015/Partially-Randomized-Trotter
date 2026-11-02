# H6確認：hash再利用・8workerへ引継ぎ済み

SOURCE `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7`、run ID `h6-confirmation-fast-handoff-v1`、coordinator PID39637。
[起動・引継ぎ契約](../../../../../docs/research/track_a_h6_confirmation_fast_handoff_v1.md)。

- [利用者の切替・1720/844上限変更承認と実行command](user_launch_receipt_v1.json)
- [exact approved grant](authorization_approved_v1.json)
- [source251/profile/input/159 reuse/残97/carry/fresh CPU/RAM/PSI/OOM/FS/inode/quota](prelaunch_gate_v1.json)
- [元run native STOP/163 worker reap/partial4証拠](parent_stop_receipt_v1.json)
- [新run native journal/worker source・CPU・thread・AS観測](launch_observation_v1.json)
- [technical sealed manifest](../sealed_v1/manifest_fast_v1.json)

159 whole groupを同じtask/seed/両軸/event/compilerで再利用し、残97（partial4を含む）を8workerで一回実行中。科学標本は8構成×32＝256のまま。探索8をpoolしない。
worker CPU20/21/24/25/26/27/28/29、coordinator22/23、内部thread1、AS32/RSS9GiB、aggregate76/jobRAM96、nice10。phase/total wall cap=null。
累積compile1720/trajectory844は明示承認済み。その他caps/carry保持、partialactual=null・別full-task予約、返却/resetなし。新確認完了や科学的PASSはまだ主張しない。
完了またはfail-closed STOP後mandatory STOP、自動retry/追加sample/H8/次stageなし。旧SOURCE/runtime/証拠とH4/venv/共有設定は保全。

read-only status： `/home/AbeHiromu/projects/h6-handoff-evidence/20261012/confirmation-fast-handoff-v1/MONITOR_STATUS.py`
coordinator log： `/home/AbeHiromu/projects/h6-handoff-evidence/20261012/confirmation-fast-handoff-v1/coordinator.log`
新runtime： `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-hash-reuse-eight-worker-20261012/artifacts/resource_applicability/track_a_h6_confirmation_fast_v1/2026-10-12/launch_confirmation_fast_v1/`
Git公開は既存認証待ち。source/認可・軽量準備資料だけをlocal commit、NPZ/runtime/cache/credentialを新stageしない。
