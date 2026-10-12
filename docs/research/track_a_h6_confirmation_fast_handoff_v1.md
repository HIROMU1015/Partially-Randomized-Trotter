# H6独立32確認：4workerからhash再利用・8workerへ一回引継ぎ

利用者が「この実装の方に計算を引き継いで」と指示し、累積compile1712→1720/trajectory840→844の変更を明示承認した。[認可・再利用・fresh gate・native起動観測入口](../../artifacts/resource_applicability/track_a_h6_confirmation_fast_preparation_v1/2026-10-12/launch_authorization_v1/README.md)。

元SOURCE73018cbcのrunはUSER_STOP_REQUESTで停止し、completed159/dispatched163/pending93、全own worker回収・cleanup errors0、coordinator PID20241はexit1をnative wrapperで観測した。manual STOPであり、旧source/成果/partial/unknown countersを保全する。旧runのexit1を新計算の障害や成功に読み替えない。

固定new SOURCE `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7`、closure251と旧245/science224/compiler/profile/input/DF/archive/stateを再照合。159 complete whole-trajectory groupのnative TASK_COMPLETE/exit0・両軸・元task/seed/event/compiler・metadata bytes/SHAを受理し、残97（停止partial159–162を含む）を元task/seedのまま固定した。各選択構成32、計256 group/512 wrapper、探索とpoolしない。

既知carry compile1506/trajectory737などに、旧歴史予約と今回partial4のfull-task予約compile8/trajectory4を別途保全した。新demand compile194/trajectory97、予約込みadmission compile1720/trajectory844。未知partialactualはnull、予約を実消費と呼ばず、返却/resetなし。承認済みexact grantにSOURCE/hash policy/reuse/tasks/profile/resources/caps/outputを結合した。準備manifestの未承認flagsは元bytesのままで、approved grantだけが一回を認可する。

fresh SOURCE/profile/input/全592探索・159 reuse/remaining seed/carryとCPU/RAM/pressure/OOM/FS/inode/quota、未使用output/control・one-shotを検査して合格。利用可能RAM約464.57GiB/disk約401.33GiB/quota KNOWN。CPUは割当可能affinity内で、kernel RAM予約を主張しない。元101人工tests・microbenchmarkを繰り返さず、sourceを変更しない軽量binding確認で起動へ接続した。

worker8 CPU20/21/24/25/26/27/28/29、coordinator22/23、内部thread1/Qiskit num_processes1、own affinity/nice10、各AS32GiB/RSS9GiB・aggregate76/jobRAM96。fresh host248.25/effective96GiB、running host168.25/effective16、host full PSI5%/全cgroup full PSI>0/新OOMでSTOP、1秒観測・最大5秒、既存own cleanupを維持。wall cap=null。他caps/科学条件/compiler optionsは不変。

run ID `h6-confirmation-fast-handoff-v1`、coordinator PID39637。8workerのnative source/CPU/thread/AS claimと処理進行、first_stop_reason=nullを観測した。これは起動確認であり、完了・科学的PASS・production speedup測定ではない。完了/STOP後mandatory STOP、自動retry/追加sample/H8/次stageなし。

共有設定/既存venv/他jobは変更せず、H4は継続。GPU query/useなし、作業/log/cacheはhome配下。新commitは軽量seal/grant/起動観測/資料だけで、raw NPZ/runtime/checkpoint/cache/credential/SSH情報を新stageしない。Git公開は既存認証待ち、local fixed SOURCEから利用者のlaunch指示に従い起動した。
