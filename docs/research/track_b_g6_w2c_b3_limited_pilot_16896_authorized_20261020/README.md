# Track B：限定Pilot 16,896件完了・mandatory STOP

**LIVE_TECHNICAL_COMPLETE_AWAITING_REVIEW**。実G7/F3で6phase/48pair/16896完了、保存値検査PASS。accepted16280、ZERO_R324、coin reject292、全correctness coverage。追加0/retry0/resumeなし/claim spent/mandatory STOP。一般科学保証・方式順位は認定しない。

- [Q1–Q10 handoff](handoff.md)、[機械可読結果](result_summary.json)、[cell/block費用](cost_tables.md)
- [固定manifest](approved_manifest.json)、[preflight](preflight.json)、[原承認](user_approval.json)、[採択契約](adopted_contract.md)
- [SOURCE/input identity](source_inventory.json)、[SOURCE保全](measured_source_retention.json)
- [人工受入](artificial_acceptance.json)、[固定checker](saved_evidence_check.json)、[追加保存値照合](supplemental_saved_checks.json)
- [費用元表](cost_and_outcomes.json)、[保存量照合](storage_reconciliation.json)、[原本inventory](raw_inventory.json)、[旧原本保全](preservation_after.json)

測定SOURCE `c7e8806cf308131b5629ac54b627d46c360799a3`、manifest `92ddf84e0d6d90ead7a98dac7eadf69ef9c2bf0622694c8870b69cf841a23bb6`。CPU1258.033384573秒/wall792.156549420秒はparent＋reaped child hierarchy（Git helper含む）の固定窓。worker own/partitionと重ねて加算しない。baseline off/instrumented on、fused両pass。人工・旧512・今回16896は別証拠。

新namespace `src/trottertracks/algorithm_codesign/g6_w2c_b3_limited_pilot_live_v1/`、focused tests `tests/tracks/algorithm_codesign/test_g6_w2c_b3_limited_pilot_live_v1.py`。SOURCE/科学Core/旧claimを変更して再実行しない。公開照合はpublication_receipt.jsonを参照。次の担当はユーザー・GPTによる後続判断。


公開照合完了：branch `track-b-g6-w2c-b3-limited-pilot-16896-publication-20261020`、payload HEAD `7e65efdb43ce691d2b9f899495339c3adf4e58d6`。公開SOURCE commit `9849aa263bd3e46321de12d70be5947f530a3c4a` のtreeは測定SOURCE `c7e8806cf308131b5629ac54b627d46c360799a3` と完全一致。API公開commitはauthor/date/parent metadataが異なるためSHAを混同しない。結果treeはlocal `1aad6c77b55c331e57d7660fb2d9c8d2a246730c` と一致。69公開payload files＋155 SOURCE pins＋10 input pinsの234 HTTP取得を全bytes/SHA256照合し、約25MBの実raw archiveも公開済み。詳細は[publication_receipt.json](publication_receipt.json)。このreceiptは上記payload後に別commitで公開する。科学実行は終了・mandatory STOP。
