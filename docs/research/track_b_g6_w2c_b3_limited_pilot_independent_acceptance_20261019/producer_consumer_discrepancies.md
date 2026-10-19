# 不一致・保証限界・最小修正要件

## SRC_TRACE_RESERVE_ACCOUNTING：再現済みproducer不整合

対象SOURCEはb9cabcfc / 公開76ca3c40の `g6_w2c_b3_pilot_candidate.py`。Coordinator.set_state wrapperはoriginal_setでstatefileを更新してからTrace.emitを呼ぶ。一方、継承scheduler.reserve_startはstates.setが戻った後にreservations、row.next、current ticketを更新する。

独立人工fixtureは最初のRESERVE trace行のfile.writeだけに一回OSErrorを注入した。SOURCE bytesは変更せず、fake endpoint/S mode/workers1/profile16を使用した。

|条件|観測|
|---|---|
|planned|768|
|state NOT_STARTED|767|
|state START_RESERVED|1|
|report reserved|0|
|report started|0|
|complete|false|
|fake claim|STOPPED|
|独立checker|INTEGRITY_MISMATCH / GLOBAL_STATE_COUNTER_BINDING|
|実科学call|0|

期待はreserved=planned−NOT_STARTED=1。未ACK/trace write例外が予約を未実行へ戻したことにしてはいけない。今回のsourceはSTOPし成功を返してはいないが、report counterと残存slotが整合していない。`producer_findings.json` と `artificial_replays/fresh_trace_reserve_write_failure/` が原証拠。

**次のproducer版で要修正。** state・counter・ticketの更新がtrace I/O例外でも保守的に整合し、失われた予約をunknown/nonrefundableとして残すこと、incomplete traceと欠けたcredit記録を明示し、ENTERを新たに許可しないことを検査する。SOURCEを結果に合わせて書き換えたり、slotをNOT_STARTEDへ戻したりする案ではない。今回固定SOURCEは変更しなかった。

最小再現は、新しいowned runtimeをfixed_reference/treeからコピーし、run_trace_write_failure.pyをそのruntimeと未存在case先へ実行する。その後独立checkerを当該新runtime identityで読む。人工source replay専用であり、実G7 bridgeやclaimを作らない。scriptには旧ローカル既定pathも残すが、CLIで別pathを指定できる。

## その他の差は欠陥と区別

- 公開18件の4 admission-only caseはsnapshot＋観測申告だけなので、独立総合判定をINSUFFICIENT_EVIDENCEとした。server側はこれらをraw完了caseとはしていない。race等の虚偽を確定したのではない。
- inventoryの2管理scriptは測定commit後の追加。主要SOURCE/schemaの測定bytes不一致ではない。
- co_filenameは任意callableのコード認証ではない。guard fragmentで無害な別callableを通せることを示したが、完全な実科学迂回を実行・証明していない。trusted-caller境界の明示事項。
- S/P1二条件bridgeと新科学seed接続は元から未実装。単条件fake projectionの成功で埋めない。
- 新Decisions readerは固定3phaseの範囲を追加検査する。旧core/旧checkerは変更せず、新しいtrace用の厳格化とした。

これらを科学的優越性・最適Nの判断へ変換していない。
