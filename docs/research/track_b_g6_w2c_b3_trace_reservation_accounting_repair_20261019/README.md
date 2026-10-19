# W2c-B3 予約会計・保存故障修正 V2

**技術修正・人工検証完了、科学入口閉鎖、mandatory STOP。** `SRC_TRACE_RESERVE_ACCOUNTING`を固定旧SOURCEで再現し、新しい人工専用SOURCEで修正した。旧Producer、独立checker、科学SOURCE/契約/結果/STOPは保全した。

|検証|最終結果|
|---|---|
|固定旧版の同じRESERVE writer故障|state予約1、report予約0。固定独立checkerがGLOBAL_STATE_COUNTER_BINDINGを検出|
|修正版の同じ故障、S/P1|reserved1、started0、開始前取消1。予約・cursor・ticketを失わずSTOP。traceは証拠不完全|
|S/P1保存fault matrix|108ケース：正常2、TECHNICAL_FAILURE40、INCOMPLETE_EVIDENCE66。全条件の期待分類・保守的会計を確認|
|今回のunit/regression|既存779＋新41＝820 PASS、最終FAIL/ERROR/SKIP各0。matrix件数をtest数へ加えない|
|保存rawの再読|108件をproducer非importで照合。固定の独立会計関数によるexact state/counter照合は82件。残りは保存証拠欠損を明示|
|canonical互換性|固定旧正常/S/P1正常の全24pairで、generator実測時間以外の全canonical payloadが一致|

[再現](defect_reproduction.md)、[原因と修正](root_cause_and_fix.md)、[SOURCE/契約差分](source_and_contract_delta.md)、[STOP/unknown](stop_and_unknown_verification.md)、[回帰](artificial_regression_results.md)、[独立checker比較](independent_checker_comparison.md)、[引継ぎ/Q1–Q9](handoff.md)。機械可読の要点はverification.json、reservation_accounting_verification.json、fault_injection_matrix.json、source_inventory.json。

Producerは`g6_w2c_b3_pilot_storage_v2.py`、追加checkerは`g6_w2c_b3_storage_verify_v2.py`。最終検証SOURCEは`45bf46b4a833e71f138903fb511f2741e54043b7`。原Git履歴はmeasurement_source_history.bundleにも保全する。公開payloadの管理commitとは区別する。

固定独立checker `49a0dcc9c8ec6dfc96eba5bb862cfdcb3d0fc866`は変更していない。V2 schema/identity全体への対応は未提供なので、最終独立受入は**後続のローカルCodex**が担当する。今回の追加checkerは主担当による補助検証であり、別人・別AIの独立reviewとは呼ばない。

raw/詳細report/logはfault_cases_final.tar.gzとそのmanifestに収録。公開時は4MiB以下のpartへ分割し、archive_parts.jsonに各partと復元archiveのSHA256を記録した。`python3 restore_artifacts.py fault_cases_final.tar.gz`で復元し、このpacketで`tar -xzf fault_cases_final.tar.gz`を行うと、verify_saved_matrix.pyが再読できる。旧再現rawはbefore_repair/。途中試行はpreliminary_artifacts.tar.gzへ区別して保存し、同じhelperで復元できる。publication_receipt.jsonは公開contentの読み戻し結果。

実G7/F3 sampling/import、新科学計算、保存科学値再解析、科学SOURCE採択、実authorization/grant/claim、quantum native build/compile/synthesisは0。production bridgeは未実装。pilot n256/二条件16,896 callsもFull-Nも実行・採択していない。
