# H6 7完了結果を再利用・残135件の一回再開

Branch `track-a-h6-roundoff-resume01-20261012`、SOURCE `78a39051d8b6d41c33d592ecfe172f8824adfaeb`。
利用者の「計算を再開して」を根拠に、新一回grantで元135 taskを実行する準備を完成。
[70限定tests](limited_tests_v1.json)、[軽量差分review](limited_review_v1.json)、[reuse/source/profile/input/carry seal](sealed_v1/manifest_candidate_v1.json)、
[利用者指示](launch_authorization_v1/user_authorization_record_v1.json)、[新grant](launch_authorization_v1/authorization_approved_v1.json)、
[fresh gate](launch_authorization_v1/prelaunch_gate_v1.json)、[command](launch_authorization_v1/launch_request_v1.json)。

450/451/452/453/454/455/457の7 group・14 wrapperを旧source0365f0a0と原bytes/SHA/task/seed/両軸/validation/compile会計で再受理。
STOP・partial456/458・消費済みgrant/lock・旧runtimeを保全。7結果を再計算せず、135件だけを元index/seed順で新exclusive出力へ計算する。
未完了456/458のpartial wrapperを流用せず、最初からsame seed/taskを取得する。Input/SCF/DF/state/signalの生成・再評価はしない。
旧449＋Phase A447＋今回旧7＝457受理/592、135件で592探索を完成する計画。完了後selection/独立32確認/H8へ進まない。

監視修正SOURCE00b8d7e3を保持。Owned PF_EXITING/identity確認、nonblocking1秒退出待ち、WEXITED後reap、欠測RSS保持/予約、host/他worker観測を維持。
科学224・基盤233・丸めbridge/cost portは不変。Floatのみ絶対1e-12/相対1e-10受理、元choice/sign/order/seedとcompiler/full-wrapper/cost定義を保つ。
今回のruntime差分はcontract/preparer/全592のreuse統合だけ。新source closure240、tests70 local PASS。独立agent review/CIではなく軽量差分確認。

Worker4 CPU20/21/24/25、coordinator22/23、内部thread1、own affinity/nice10。
AS32GiB/worker、sampledRSS9GiB、coordinator soft2/hard32、aggregateRSS40GiB、jobRAM64GiB、wall cap=null。
5秒監視・元PSI/OOM・容量/inode/quota・output/log/instruction capsを維持。H4はRUNNING/807signalsを読取確認し、変更0。
共有環境・venv・他job・GPU query/use・追加準備transpileなし。

Known carry compile918/control292/trajectory443/occurrence1096/primitive1470/reference400。
旧中断の実総数はnullを保持。直近partial456/458の全登録task ceilingを独立予約compile4/control4/trajectory2/occurrence16として扱う。
さらに前の4dispatch STOPの8compile slotsとtrajectory4/occurrence32を保守的予約。予約はactual totalではない。
合計予約compile12/control4/trajectory6/occurrence48、残135 demandcompile270/control68/trajectory135/occurrence1080。
Known＋予約＋新demandのadmissionはcompile1200/control364/trajectory584/occurrence2224、元1712/1024/832/6656以内。
不明の古い歴史をゼロ扱いせず、実全歴史のexact totalと呼ばない。失敗分返却・budget reset・cap拡張なし。

SOURCE/profile/input/reuse7/carry/fresh CPU/memory/PSI/OOM/disk/inode/quota、未使用output/controlをPASS。
Output `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-roundoff-resume01-20261012/artifacts/resource_applicability/track_a_h6_roundoff_resume01/2026-10-12/launch_parallel_v1`。
Run ID `h6-roundoff-resume01-135-launch_parallel_v1`。ログ/終了証拠 `/home/AbeHiromu/projects/h6-handoff-evidence/20261012/roundoff-resume01-v1`。
資料commit時点READY_NOT_STARTED。実際の起動は同所coordinator.pid/native execution_claimを参照。
計画flags approved=false/runtime_authorization=false/allowed_cpus=[]。新exact grantだけが今回一回を認可する。
完了/STOP後終了、自動retryなし。過去partial resume禁止のflagsは維持し、complete-only reuseで新workersを起動する。
User push方針を保持、agent push0。最新明示指示を根拠にlocal固定SOURCEで起動する。

利用者push:
```bash
git -C /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-roundoff-resume01-20261012 \
  -c maintenance.auto=false -c gc.auto=0 \
  push origin HEAD:refs/heads/track-a-h6-roundoff-resume01-20261012
```

## 一回起動確認

[Native起動receipt](launch_authorization_v1/started_receipt_v1.json)。CoPID 4101124、先頭456/458/459/460を4workerで実処理中。保存7結果はdispatchしない。起動時STOP理由null。
Read-only status: `/home/AbeHiromu/projects/h6-handoff-evidence/20261012/roundoff-resume01-v1/MONITOR_STATUS.py`。完了/STOP後終了。

## 完了・結果集約

上のREADY/起動中記載は当時の履歴です。新135が完了、保存7と合わせ142/142、全探索592/592・1188 wrapper。
[最新の結果入口](../../track_a_h6_exploration_summary_v1/2026-10-12/README.md)。exit0、cleanup error0、wall約4時間18分、aggregate peak11.1797GiB。
独立32確認は未実施。新計算/grant/pushなし、原source/runtime/旧証拠は保持。
