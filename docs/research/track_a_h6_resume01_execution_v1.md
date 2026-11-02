# H6 complete-only再利用による残135探索の一回再開

[一つの資料入口](../../artifacts/resource_applicability/track_a_h6_resume01_preparation_v1/2026-10-12/README.md)。
SOURCE78a39051d8b6d41c33d592ecfe172f8824adfaeb。利用者が再開を指示。
旧7完成450/451/452/453/454/455/457をbytes/SHA/元task/seed/両軸/source/validation/callで再受理し、原142から7件だけ除いた135へ固定。
旧科学224/基盤233/丸めbridge/compiler/own資源上限は不変。修正済みkernel退出監視を保持。
Incomplete456/458のpartial費用を流用せず、新workers/output/one-shotで同じtask・seedを最初から取得する。
不明actualはnull、過去消費は返却/resetせず、完成確定値と未完了task ceilingの独立予約を明示。
Known＋予約＋新demandはcompile1200/control364/trajectory584/occurrence2224で元caps内。全歴史のexact totalを主張しない。
70 local限定検査PASS、source/profile/input/保存7/Fresh resource/output/control PASS、新利用者指示bound grant。
CPU4workers20/21/24/25・coordinator22/23、AS32/RSS9/aggregate40/RAM64・thread1・wall cap=nullを維持。
旧449＋Phase A447＋7保存結果＋135新取得の原592集合を検査して探索完了とする。selection/独立32/H8へ進まない。
完了/STOP後終了、自動retryなし。共有環境/venv/H4/他job/GPUを変更せず、user push方針を保持。

2026-10-12 一回起動確認：coordinator4101124、元135の先頭456/458/459/460が4workerで実処理中。新SOURCE78a39051、起動REVIEW4107286a。保存7結果は再計算せず、監視下で完了/STOP後終了する。

## 2026-10-12 完了後の保存結果集約

[探索結果・限界・一つの資料入口](track_a_h6_exploration_result_v1.md)。
135/135を取得、旧7と合わせ142/142補完、旧449＋Phase Aと統合して592/592・1188 wrapper。
原terminalはH6_PARALLEL_COMPLETE_MANDATORY_STOP、reason=null、coordinator exit0。新135 wall15460.684819秒、aggregate peak12004102144 bytes、135全wait/reap・cleanup error0。
保存scalar/JSONを再受理し、全精度のshot込みmapを集約。新科学計算/確認/grantなし。独立32確認未実施、population winner未認定。
