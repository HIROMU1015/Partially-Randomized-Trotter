# Track B W2c-B3：サーバー内独立技術受入

**人工技術受入完了。実科学未採択、production gate閉鎖。Mandatory STOP。**

検証対象HEAD `80b7bf61fe0a735d4031bb719c4b3e788611c3ba`、測定SOURCE `b7ad344401b10395c49e4cfa806909282859bab5`、artifact `024d8812e97c5d541391533680fdbb3aee4cf6d8e51f020d3302a00767af42cb`。
独立基点 `5c1e13fa74b3c5e4f3166babed785274844655d0`。branch `track-b-g6-w2c-b3-real-adapter-independent-server-20261020-r2`。固定Producerのmerge/rebase/修正はしていない。

- 新独立check **224 PASS**：Codec49、Result/Context/Angle/Audit148、Bits16、parent gate1、fake authority10。
- 親/spawn workerのproduction拒否 **30 calls**は別集計。guard記録57 process、禁止core科学moduleのload/import試行0。
- 保存人工22 case：正常2、技術失敗13、証拠不完全7。semantic mutation20件は独立に全件拒否。
- n256全16,896 trial完了/join一致。full1,536・hash-only15,360。未保存原document直接取得は主張しない。
- 最終SOURCEでProducer unit102/integration30と旧差分回帰107を再実行し、全PASS。新scopeの人工Bundle2件（正常n16/Angle失敗）も独立検証。
- SOURCE89/input10、科学8 module/201 entryの静的照合、archive994 member、content82 file、旧291 SOURCE/test bytes一致・保全。

[Q1〜Q10](handoff.md)、[機械判定](verification.json)、[公開照合](publication_receipt.json)、[次の担当](pending_and_required_decisions.md)。
旧独立1,038・Producer132・旧107・今回224・20 mutation・22保存case・追加2人工caseは合算しない。

当初の§3Eはdriver/helperも静的限定だったため、依存をload前に遮断し保留した。
その後の利用者の明示修正で、この2 moduleの人工用途だけを許可して残試験を完了した。
[範囲修正と元の依存経路](import_scope_finding.md)・`*_prior_scope.json`に旧判断を保持する。

## 独立性・再現

別Codexセッション・新worktree・独自parser/canonicalization/hash/Result oracleを使用。AI sub-agent0、物理host独立性は主張しない。
`independent.py`は候補Codec/Result/Fixture/Offlineをimportしない。既存独立Storage V2 consumerは保存・STOP・会計だけへ再利用。
`candidate_probe.py`は別processのsubjectであり、期待値生成には使わない。

`audit_source.py`→`run_audit.py`が固定Git archiveと22case/20mutationを検証。
`run_probes.py`が214独立probe、`authority_probe.py`が新FAKE_ONLY scope、`run_fixed_tests.py`がProducer132/旧107を別集計。
`fresh_bundle.py`が親/worker preload guardと新規人工2scope、`static_audit.py`が固定差分・保全を確認。
`finalize_report.py`は利用者修正後の最終判定を生成する。`build_report.py`は元scopeでの保留記録のgeneratorとして残す。
原rawの再取得には固定Producer archiveを使い、新rawは[fresh archive/manifest](fresh_evidence_manifest.json)に保存した。
一時runtime `/tmp/b3-independent-server-r2-20261020/runtime` は固定blobから構成し、再実行では新scope/outputを使う。
