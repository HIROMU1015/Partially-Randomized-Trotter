# A1 N3：再発防止と観測性の改善

今回の直接原因はUNKNOWN。[診断](N3_INTERRUPTION_DIAGNOSTIC.md)を前提に、特定原因の修正完了とは称さない。
旧科学SOURCE03928・旧397 bit/価格・凍結係数・STOPは不変。以下は技術提案と人工検証で、科学実行へ未接続である。

## 原因候補への対応と、原因を記録する対応

| 対象 | 実行を保護する案 | 観測を改善する案 |
|---|---|---|
| 会話/SSH/tool切断 | 科学processを会話sessionの直接の寿命から分離し、再接続は監視だけにする | manager ID、PID/start ticks、boot、PGID/SID、永続job IDを開始時に記録 |
| 親のsignal/異常終了 | 独立managerの下でRestart=no、最初の失敗で新trial投入停止 | manager側のraw wait status、signal、service Result/ExecMainCode/Statusを別fileへ保存 |
| worker異常 | 取得bitをfsyncしてから価格処理、未取得と未価格を分離 | worker PID/start/end・生存状況・exitcode/例外型・最後のcall/gate位置を記録 |
| OOM/cgroup/競合 | 実際の利用可能CPU、cgroup headroom、PSS/USS等を見て自分の新規投入を調整 | 実childのcgroupと全ancestorのmax/high/current/peak/eventsを開始前・途中・終了後に保存 |
| 正常例外/STOP | STOP_REQUESTEDをpool drainより先に耐久保存。再試行せず既存tail回収 | UTC+monotonic、CALL_START/RETURN/EXCEPTION、部分行のhashとUNKNOWN分類 |
| manager自体の消失 | 既存の承認済み持続manager/batch基盤を使う | manager外に残るjob終了記録。managerとchild全体のSIGKILLをchildだけで保証しない |

SIGKILLのhandlerは作れない。signal番号が分かっても送信元は別情報であり、wait statusだけでOOM/利用者操作等を断定しない。
PythonのPOSIX raw returncode=-Nはsignal N、通常exit137をそれだけでSIGKILLと推定しない。[subprocess仕様](https://docs.python.org/3.12/library/subprocess.html)
memory.eventsは階層counter、memory.events.localは当該groupのcounter。現在値だけでなく、同じchild membership・ancestor・bootを結びつけた差分が必要。[Linux cgroup v2](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html)

## 今回追加した最小コードと再利用

- [journal診断](../../../src/trottertracks/algorithm_codesign/a1_n3_interruption_diagnostics.py)：archive/member identity照合、durable bit/価格・未完了call分類、末尾の不完全行と途中破損の区別。旧`reconcile`の関数ASTだけを固定bytesから再利用し、科学moduleをimportせず別scratchで会計を照合した。
- [lifecycle prototype](../../../src/trottertracks/algorithm_codesign/a1_n3_lifecycle_observer.py)：自分のdirect child一回だけのPopen/wait、shellを介さないraw終了値、PID identity、UTC/monotonic、実child cgroup/ancestor snapshot、結果を含めないcall start/return/exception。
- [技術runner](../../../scripts/diagnostics/a1_n3_interruption_audit.py)：保存archiveを読む入口のみ。N3起動やresumeのCLIはない。
- 既存`parallel_validation_executor.atomic_write_json`と`shared_cpu.module`のimport境界を利用。元executor本体のattempt/resume・thread固定・resource admissionは旧N3契約へそのまま合わないため起動しない。既存科学source/serializerも書き換えない。

prototypeはN3親/77workerへの接続済みsupervisorではない。direct childのwait結果は取得できるが、grandchildのwait、observer自身のSIGKILL、永続managerの寿命は保証しない。
全worker exit statusを得る次driverでは、親が所有するProcess/managerの終了記録を保存し、非子processへ別途waitpidする競合を作らない。
既存ProcessPoolExecutorのprivate属性へ依存するなら、固定Python版で人工的なabrupt worker deathを検証し、その依存を明記する。監視PIDの消失だけをexitcode取得と呼ばない。

## 人工テスト

[19件PASS](../../../artifacts/o2_a1_n3_interruption_diagnostic/2026-11-06/TESTS.json)、[raw log](../../../artifacts/o2_a1_n3_interruption_diagnostic/2026-11-06/TESTS.txt)、[tests](../../../tests/test_a1_n3_interruption_diagnostics.py)。
invented journalと短命な自分のPython childだけを使用した。SIGTERM/SIGKILLはtest childが自分自身へ送信し、共有serverや他jobを操作していない。

確認範囲は、exit7とexit137・signal終了の区別、child一回/heartbeat/no timeout-retry、衝突時の未起動、clock/identity、childとmonitorで異なるcgroupの区別、call return/例外、bit先行保全、重複/価格不一致/末尾破損、固定reconcile再利用、科学依存の非import。
初回18件中1件で通常`trotterlib.__init__`による不要な科学依存importを検出した。既存shared_cpu経路へ修正し、最終19件PASS。初回FAIL logも保全した。
実state、Born、RNG sampling、測定、compile、variance/Nは0。科学的CV改善や実N3完走・外部CIの証拠ではない。

## 次回のprocess管理方法

現在SERVER1ではsystemd-run/tmux/nohup/setsidが利用でき、Slurm/PBSの起動commandは見つからなかった。user managerはrunning、**Linger=no**だった。
第一選択は、既に許可され寿命・終了記録が保証された持続manager/batch基盤の利用である。利用可能ならN3専用jobとして一回起動し、会話からはjob IDの監視だけを行う。

user systemdのtransient **service** は候補となるが、user managerの永続性が前提である。Linger=noでは最終logout後も生存する保証を置けない。[systemd-run一次資料](https://github.com/systemd/systemd/blob/v255/man/systemd-run.xml)、[loginctl v255](https://www.freedesktop.org/software/systemd/man/255/loginctl.html)
現在のSSH session内scopeへ起動するだけでは分離にならない。Restart=noとし、終了理由を回収するまでunitを自動収集・消去しない運用が必要。
tmux/setsid/nohupは端末・SIGHUPへの対処候補だが、session/cgroup全体のkillやhost停止を防がない。これら単独を全中断の解決策としない。

今回はunit/jobの起動、linger設定、logind/system/cgroup/quota変更を行っていない。永続性が現在のユーザー権限と既存運用内で確保できなければ、未解決の実行前条件として残す。
新しい承認証明・grant発行やsudoをユーザーの科学実行許可の要件として追加しない。

## 次SOURCEに必要な接続（今回は未実施）

1. 旧exact kernel・law/policy/state・gate phase/順序・53-bit規約はそのまま、新しい技術adapterだけを固定する。観測結果にstate/Born/真信号を含めず、CVへ渡さない。
2. trialのlogical IDとattempt IDを分離し、parent/worker/source/domain/ordinalへ結びつける。bit・価格receiptはatomicに確定し、重複を拒否する。file fsyncに加えrename先directoryの耐久性も明記する。
3. parent/workerにsignal/exit記録、call return、定期的なgate位置とresource情報を追加する。確率・巨大stateのdumpをheartbeatにしない。
4. 最初のSTOPをdrainより先に保存。新規投入停止→取得bit保全→価格/未知を分類→tail回収とする。終了handlerだけを唯一の保全経路にしない。
5. 起動前と実行中に実際の空きCPUとメモリ、cgroup ancestorの制約、pressureを読む。固定の恣意的core/RAM/worker数を追加しない。他者を圧迫する場合は自分の新規投入を控える。観測RSSを厳密上界にせず、他process/共有設定を変更しない。
6. 本番起動前に、同じmanager経路で人工jobの親終了・worker異常・session切断と記録回収を検証する。現在のprototype PASSはこの配備検証の代わりにならない。

これらは科学実行許可ではない。改善SOURCEの技術接続・人工確認・固定と、案A/Bそれぞれの有限科学scopeの明示許可が揃うまでN3を起動しない。
