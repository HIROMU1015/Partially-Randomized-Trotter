# Global Bundleへの未接続境界

作業基点 `6bffb5b5819918698b522270ff19b255766d3072`。API候補公開HEAD `82896d53fa97235fa848d6c0833eb193af8b99d6`、人工測定SOURCE `78c571005fe766c61b32dc867e006531099ff608`。科学SOURCE `76ca3c4089b201d55da0fede08e1f77733ac38a5` はbytes/ASTだけを読む。これはローカル技術受入であり、科学的採択・本番実行許可ではない。

**A：新port→固定pair engineの人工接続。B：変更していないGlobal Bundleの人工回帰。A＋BのPASSは、新port入りGlobal Bundleのend-to-end PASSではない。**

静的確認：exercise.pairはArtificialEndpointを直接作ってD.run_pairへ渡す。exercise.runはphase単位のPoolを使い、port sidecar/worker/report/callerを保存する。固定Bundle bridgeの実行経路では固定人工fixtureが生成され、外部endpoint factoryを受け取る公開注入口がない。新ports.pyを参照する呼出しもない。今回、injection・monkeypatch・scheduler再実装はしていない。

[固定Bundle回帰](fixed_bundle_regression_verification.json)：公開43ケース＝完全Bundle2、condition-only2、typed technical failure13、証拠不足26。保存mutation35＝不整合32、証拠不足3。portの4 caseおよび22 mutationとは別に保存する。

固定Bundle側ではglobal claim/call/byte budget、RAM intent/physical state/trusted prefix/UNKNOWN、未ACK非返却、STOP後ENTER禁止、inflight drain、summary/receipt/diagnostic/marker/close/delivery/callerを独立consumerで結合する。不足26件は完全受入にしない。これらの保証を新portのsidecarやphase Poolへ移したとは表示しない。

将来必要な技術変更案（未承認）：別versionのendpoint/bits factory注入口、実API result/trace/codec/F01の接続、全global permit/STOP/UNKNOWN/storage/callerへportを結合、runtime/authority/candidateを再固定。これには、sidecar保持予算と本番の実authorizationを含む契約判断が要る。現在のSOURCEをbooleanで解除する手順ではない。
