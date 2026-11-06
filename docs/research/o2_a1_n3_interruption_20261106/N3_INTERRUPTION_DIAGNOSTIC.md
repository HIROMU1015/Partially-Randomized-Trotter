# A1 N3中断の技術診断

2026-11-06 JST、SERVER1 GPUA100-gleap-01。**終了原因・signal・送信元・終了コードはUNKNOWN。**
実行管理セッションの連続性が失われた証拠は得たが、campaignを終了させた直接原因との因果関係は確定できなかった。

## 固定資料・今回の境界

科学SOURCE `03928ce55bc5a1bfd070aa9f0a6bf8a990a0ff25`、中断結果 `d97e916526f431280e687815d3e970bbf03a077e`、結果を知らず固定された判定基準 `a53c1d6eedea0d85fb484425db9012aa89738129`を確認した。
判定基準の[保存コピー](references/N3_RESULT_DECISION_FRAMEWORK.md)は元blob/bytesのままで、今回書き換えていない。これはN3開始前の登録を意味しない。

対象は保存linear H4/1.00 Å/STO-3G/legacy DF12、8 system qubits・全4電子sector、L_D3、m0/q1/δ=.02/r1/K2、coefficient_atol0、X/Y、N1 actual Q/selected policy・N2保証ground・認証済exact engine。
今回は保存journal/価格/実行時間metadataの技術的照合と、読み取り専用OS・セッション調査、人工テストだけを行った。
DF/state/Born/真信号の新評価、RNG、測定、compile、fit、variance、N、科学retryは0。旧bit・価格・係数・SOURCE・STOPを変更していない。

証拠の入口は[artifact SOURCE](../../../artifacts/o2_a1_n3_interruption_diagnostic/2026-11-06/SOURCE.json)と[EVIDENCE_MANIFEST](../../../artifacts/o2_a1_n3_interruption_diagnostic/2026-11-06/EVIDENCE_MANIFEST.json)。
元の[REPORT](../o2_a1_n3_science_20261106/REPORT.md)、INTERRUPTION_OBSERVATION、INTERRUPTED_TRIALS、CALL_COUNTS、RESOURCE_START、RUN、全79 journalを照合した。

## 原因候補の判定

| 候補 | 得た証拠 | 判定・限界 |
|---|---|---|
| 親・worker終了 | 親2698500・shell2698496・77 workerが不在。旧管理親2650356も現在不在。旧session99424は取得不能 | 消失は確認、終了順・wait statusはUNKNOWN |
| 実行管理/SSHセッション断絶 | 旧管理process UUIDはpid2650356、後続はpid2770025。現在SSH scopeは20:40:40作成。20:17にusage_limit_exceeded、20:20:50に別管理processのWebSocket reset | 管理の連続性喪失は確認。WebSocket reset/利用上限/SSH切断のどれが子を終了させたかは未確定 |
| OOM/メモリ不足/cgroup | 旧親RESOURCEの最後のMemAvailableは967,023,188 kB。現在のuser ancestor memory.eventsは0、memory.max/highはmax | 観測時のhost空きは多いが最後の瞬間の不足・別cgroup・外部OOM処理を排除できない。現在値は履歴の証明ではない |
| SIGTERM/SIGHUP/SIGKILL等 | signal記録・終了コード・送信者auditなし。kernel/auth/syslogは読取不可 | UNKNOWN。終了コード137すら取得していない。特定signalを推定で採択しない |
| Python例外/worker異常 | 全journalにWORKER_STOPなし、RUNにSTOP/tracebackなし、正常finalizer成果物なし | 保存された例外証拠なし。親自体のsignal終了、native crash、未flush/後処理失敗等を排除できない |
| Exact Fraction/並列負荷 | 完了397件のengine経過時間は約307–482秒/件。旧psには高CPU使用worker。77 process、保存済RSS highwaterあり | 重い実装で監視継続・長時間運用が必要。時間の長さやRSSだけで終了原因とは判定しない |

## 時系列：JST（UTCは9時間前）

| 時刻 | 記録と確度 |
|---|---|
| 19:38:57 / 19:39:34 | 主sessionのtool出力にsession_id99424。保存bundle・RESOURCE_STARTも取得開始時の証拠 |
| 19:41:24 | ps出力で管理親2650356→shell2698496→Python2698500の親子関係とworkerを確認 |
| 19:55:17 | 会話turnのinterrupted。後続19:56:43のpsにもPython/workerが存在。このturn中断だけを今回の終了原因としない |
| 20:11:01 | 元mtime記録上、A1 X train・FROZEN保存。rank4、既定range guard合格という旧結果を維持 |
| 20:17:13.285 | 主sessionのtask_complete.error=`usage_limit_exceeded`。これは会話側エラーで、科学processのwait statusではない |
| 20:17:19 / 20:17:46 | 旧管理pid2650356のremote-control loopがexit/start、shutdown_requested=false。loop終了をOS process終了と同一視しない |
| 20:19:33.742–744 | 保存された元mtimeで最後のword tapeとworker追記。workerの最終entry時刻であり、bit保存時刻ではない |
| 20:20:35.781–782 | 元親JOURNAL/RUNの最終mtime。親journal最終行はRESOURCE、RUN末尾はY train completed77/pending77 |
| 20:20:50.351951 | 管理process **pid6417** のWebSocket reset。これは旧管理親2650356とは別のprocess。時刻近接だけでsignalやkillを帰属させない |
| 20:40:40 / 20:41:16 | 現在SSH scopeの作成時刻 / 新管理pid2770025のquery内最初のlog。旧scopeを特定した記録ではない |
| 20:44:23 / 20:44:30 | psで親不在を確認、session99424はUnknown process id |
| 20:45:39 | INTERRUPTION_OBSERVATIONの元観測で親・全77 worker不在、元STOP/ACCOUNTINGも未作成 |

元journalには行ごとのUTC/monotonic時刻がない。上表のjournal時刻は中断結果のINTERRUPTION_OBSERVATIONが保存した**元mtime**であり、新checkoutやtarのmtimeではない。
archiveのmtime=0は圧縮時の固定metadataなので時系列へ使わない。別worker間の厳密な全順序、最終bit/価格の絶対時刻、processの終了瞬間は復元できない。
SQLiteのprocess別first/lastは限定query範囲内のlog境界であり、processの生没時刻ではない。
証拠：[SESSION_TRACE](../../../artifacts/o2_a1_n3_interruption_diagnostic/2026-11-06/SESSION_TRACE.json)、[RUNTIME_LOG_EXTRACT](../../../artifacts/o2_a1_n3_interruption_diagnostic/2026-11-06/RUNTIME_LOG_EXTRACT.json)。認証情報や会話全文は公開していない。

## journalの確定事項

[JOURNAL_DIAGNOSTIC](../../../artifacts/o2_a1_n3_interruption_diagnostic/2026-11-06/JOURNAL_DIAGNOSTIC.json)と再利用した[RECONCILED_CALL_COUNTS](../../../artifacts/o2_a1_n3_interruption_diagnostic/2026-11-06/RECONCILED_CALL_COUNTS.json)は保存結果と一致した。

- 親planned/path-fixed/worker-startは各474。保存bit397、価格397、親TRIAL_COMPLETE397。重複bit・bit前の価格・bit/価格のrecord不一致はない。
- A1 Xは五層各64の320件、Yは層0=64・層1=13の77件。cal/productionは未着手で、科学的な平均・精度・比較を再評価していない。
- 未完了77は全てA1/Y/train。層1が51、層2が26。各々CALL列は`exact_engine → selected_full_build`で、Born入口・bit・価格の記録なし。未完了のbit/価格を0で埋めない。
- 親journal内で最後の正常completionはY/層0/trial25、ticket=`5c9666f2ace9d0cfd1c86c3638262231754e95555053fa8fec0172d8a37dbdfb`。完了順はtrial番号順ではない。
- 最後の親行はRESOURCE（load約80.67/75.75/84.84、上記MemAvailable）。全worker末尾を含めて397価格との整合を確認した。
- malformed/truncated tailとWORKER_STOPは全79 memberで0。これだけで未記録のsignal/例外がなかったとは言えない。
- 関数入口記録はexact_engine474、selected_full_build871（測定/価格を含む）、Born397、compile206。呼出完了回数とは区別する。

### `selected_full_build`の意味

旧driverの`counted`は`SavedEngine.selected_circuit`の**呼出直前**にfsyncする。
`SavedEngine.probability`はその後、列の検査・stateのprivate copy・各gateのmatrix照合と`apply_exact`を行い、最後にBornを呼ぶ。
build終了・各gate開始/終了のjournalはない。したがって未完了77件は「buildへ入った」までは確定するが、停止時にbuild内だったか、build完了後のgate列だったかはUNKNOWN。
未完了callを「build失敗77件」や「Born失敗77件」と分類するのは誤りである。

## OS証拠の取得範囲

kernel/user journalを10:35–11:50 UTCで照会したが、取得行は0。list-bootsは「system/他userのmessageを閲覧していない」と通知し、可視journalの最終日時も事件前だった。
dmesgはOperation not permitted、kern.log/syslog/auth.logはPermission denied。sudoや権限・共有設定の変更は行っていない。
従って「OSにOOM/kill記録がない」という否定証明には使えない。
現在scopeは`/user.slice/user-30038.slice/session-212829.scope`、20:40:40開始。旧workerのcgroup membership/上限/counter差分は未保存。
現在の0 countersや豊富な空きRAMを旧workerの最終状態へ移植しない。
参照：[OS_LOG_ACCESS](../../../artifacts/o2_a1_n3_interruption_diagnostic/2026-11-06/OS_LOG_ACCESS.json)、[OS_FILE_ACCESS](../../../artifacts/o2_a1_n3_interruption_diagnostic/2026-11-06/OS_FILE_ACCESS.json)、[SESSION_OS_CONTEXT](../../../artifacts/o2_a1_n3_interruption_diagnostic/2026-11-06/SESSION_OS_CONTEXT.json)、[現在snapshot](../../../artifacts/o2_a1_n3_interruption_diagnostic/2026-11-06/CURRENT_RESOURCE_SNAPSHOT.json)。

## 終了理由が残らなかった実装上の不足

1. foreground科学commandは実行tool/sessionに属し、永続した独立supervisorのwait結果を保存していなかった。session番号はOS終了理由の台帳ではない。
2. 親はBaseExceptionをcatchするが、SIGTERM/SIGHUP用handlerとsignal journalはない。workerはExceptionのみ。SIGKILLはhandler/finallyで保存できない。[Python signal](https://docs.python.org/3.12/library/signal.html)
3. workerの通常Python例外ならWORKER_STOP→親STOPの経路があるが、abrupt worker deathは親futureのBrokenProcessPool等となり得る。親も失われた場合にworker wait statusを回収する外部記録がない。[Python futures](https://docs.python.org/3.12/library/concurrent.futures.html)
4. STOPのfsyncより先に`pool.shutdown(wait=True,cancel_futures=True)`を待つ。shutdownの停止・例外で最終STOPまで到達しない可能性がある。これは静的に確認した弱点で、今回それが起きた証拠ではない。
5. PID/start ticks/PGID/SID/boot/cgroup、開始前後のmemory.events、call return、timestampを記録していない。RSSの保存も価格完了時中心で、中断末尾は欠測。
6. 行fsyncとbit先行保存は397件を保全できた。一方、signal後の回収・部分行の扱い・manager自身の消失を検証する人工テストと独立記録が不足していた。

## Exact Fractionと並列実行

既存397件のengine経過時間のmin/median/maxは306.892/365.196/481.624秒。これは保存timing metadataの集計で、CPU時間や新benchmarkではない。
旧codeでは各gateでFractionの積和と大整数の約分を繰り返す。浮動小数の固定単価と違い、分子分母のbit長・列長で費用が変わる。fork共有も各workerのstate更新やcacheでprivate pageへ変わり得る。
保存RSS highwaterの和を実同時物理peakや全supportの必要RAM上界には使えない。新しい実stateプロファイリングは実施していないため、gate段別の時間・bit長の増大・最終RAMはUNKNOWN。
長時間/高並列とsession依存管理の組合せは改善すべき運用上の弱点だが、Fraction、77worker、メモリ不足を今回の原因と断定しない。

**判定：技術調査は完了、直接原因はUNKNOWN。再発防止案・復旧案を別文書へ示し、新科学実行なしで公開照合後mandatory STOP。**
