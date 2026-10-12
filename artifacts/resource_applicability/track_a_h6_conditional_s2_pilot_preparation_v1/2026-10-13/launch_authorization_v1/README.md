# H6条件付きs2 pilot：一回実行承認

利用者の「次の作業に進んで大丈夫です」を、直前に提示した固定SOURCE・条件付き8 trajectory/16 compile・累積上限1736/852・提示済みCPU/memory/STOP条件の承認として記録。SOURCE **843265b624eac8fbb9acea47ef21cab1207e1880** は変更しない。旧592探索/256確認/科学224/runtime251/profile/input/既知消費・予約を保全する。

[新exact grant](authorization_approved_v1.json)、[利用者指示とのbinding](user_approval_receipt_v1.json)、[SOURCE/input/profile/carry/fresh preflight](preflight_v1.json)、[絶対argv・run ID・private証跡保存先](approved_launch_binding_v1.json)。grant SHA `f91dc3d1921d0c74cae74599699221527d1c31be77ea2b343974a4fca3df625a`。準備manifestの未承認flagはそのまま固定し、この新grantを実行認可として用いる。

worker1 CPU20/thread1/AS32GiB/RSS9GiB、coordinator22/23/ASsoft2-hard32/RSS2、aggregate40/RAM64。既定の5秒以内監視、pressure/OOM/headroom/output/call/instruction guard、own-run identity限定STOP・cleanup/reapを維持。ログはhome配下の新証跡directory、coordinatorを待つ小さなstdlib helperが新たなnative exit codeと時刻を記録する。helperは同じown-run CPU22/23、AS256MiB/file16MiB（子spawn後に適用）で科学処理をしない。

このcommit時点では未起動。remote確認後にrunner自身が再度fresh gate・未使用output/lockを検査し、一度だけ起動する。実費用結果や完走はまだ未評価。自動retry/resume・入力再生成・H8・次science stageを認可しない。完了またはfail-closed STOP後に終了し、保存metadataだけで寄与を整理してmandatory STOPする。H4/共有/venv/他jobは変更しない。


## 実行後の追記

承認runは8/8・16/16完了、native exit0・全8reap/cleanup0・残存0。[結果入口](../../../track_a_h6_conditional_s2_pilot_result_v1/2026-10-13/README.md)。固定SOURCE/grant不変、一回runのgrantは消費済み。初回helper CPU-affinity preflight拒否（科学call/lock0）も[別receipt](prelaunch_rejection_receipt_v1.json)に保持。追加実行は認可せずmandatory STOP。
