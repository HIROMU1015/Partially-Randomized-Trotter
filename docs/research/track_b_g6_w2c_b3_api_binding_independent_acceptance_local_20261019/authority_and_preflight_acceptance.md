# Authority・preflight

作業基点 `6bffb5b5819918698b522270ff19b255766d3072`。API候補公開HEAD `82896d53fa97235fa848d6c0833eb193af8b99d6`、人工測定SOURCE `78c571005fe766c61b32dc867e006531099ff608`。科学SOURCE `76ca3c4089b201d55da0fede08e1f77733ac38a5` はbytes/ASTだけを読む。これはローカル技術受入であり、科学的採択・本番実行許可ではない。

独立保存checkerはexpected bindingを独立に書き下す。candidate source release/input reference/runtime/contract digest、scope/claim、16896/64GiB/256MiB、seed、retry0/resume false、worker policy、storage destination、monitor/STOP hashを全field/typeで比較する。formal_source_release=null、real_grant=null、real_authority=falseを維持。

保存FAKE_ONLY SQLiteは `mode=ro&immutable=1` で読む。STOPPED/spent=1、CLAIM_COMMITTED→STOP_COMMITTEDという保存historyとfake_snapshot一致を確認する。これは未保存の全atomic履歴やpower-loss durabilityの証明ではない。

別processで固定Producerの人工authorityだけを制御試験：[direct controls](regressions/direct_closed_controls.json)、[原store](regressions/direct_authority_raw/)。新規private `/tmp/b3-bundle-artificial-API-independent-*` の3 scopeを使用。

- 32同時claimでwinner1/loser31。
- STOP後の二重claimを拒否、spentを維持。
- commit後ACK例外を注入するとSPENT/1、再claimを拒否し、その後STOPPED/1。
- binding全20 fieldの個別改変を拒否し、claimは未消費のまま。

実認可は発行・消費していない。fake approvalは人工tokenであり署名済み実grantではない。実store所有/鍵/失効、全worker lineage、durabilityはpending。

preflightは62 SOURCE/10 input/AST digest/executable/library/candidateの一致を要求する。ローカル人工runtimeを固定bytesから疎materializeし、新runtime identityを作ってpositive preflightを確認した。サーバー3.12のcandidateを3.10の実runtimeと同一と扱わない。独立contract negativesではSOURCE/runtime/signature/schema/input/digest/seed/adoption/retry/resume等の改変を拒否。

通常の独立offline経路はProducer authority/staticをimportしない。direct controlsとProducer73 testsは別process・別出力の補助証拠。
