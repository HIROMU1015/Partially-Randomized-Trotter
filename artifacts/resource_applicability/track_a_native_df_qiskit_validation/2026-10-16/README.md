# Native DF Qiskit接続検証 — 人工技術証跡

[主要資料・SOURCE対応・不一致分類](../../../../docs/research/track_a_native_df_qiskit_connection_validation_20261016.md)。

- [connection_validation_v1.json](connection_validation_v1.json)：最終42 tests（旧回帰27＋接続15）PASS、134 observation、実Qiskit 1.3.0／gate定義hash／参照4＋共通24＋現counter1のsource closure。
- [baseline_discrepancy_v1.json](baseline_discrepancy_v1.json)：旧1ea42fbf…カウンタの明示zero RZ除去とactual appenderの不一致。人工metadata限定、元builder不変。
- [connection_validation_attempt01.json](connection_validation_attempt01.json)：whole-partial unitary追加前の42 PASS。旧試行を保全し、最終source hashとは区別。
- [no_qiskit_fallback_v1.json](no_qiskit_fallback_v1.json)：system Pythonの27 stdlib tests。接続判定はPARTIAL_STDLIB_ONLY、最終42件へ加算しない。
- [technical_audit_v1.json](technical_audit_v1.json)：元worktree・旧source・結果・manifest履歴の保全。

原Qiskit高水準命令・固定分解費用・transpile後費用は別。今回transpileは0。
一般identity phase期待値の0〜1 P/occurrence幅は維持。
実分子計算・sampling・保存H6/H8再解析・server操作・新grantは0。MANDATORY_STOP。
