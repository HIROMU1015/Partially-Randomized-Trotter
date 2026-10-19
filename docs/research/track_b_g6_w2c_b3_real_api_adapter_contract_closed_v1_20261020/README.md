# Track B W2c-B3 閉鎖G7/F3 Adapter候補 V1

**人工技術自己検証完了。実科学入口は閉鎖。Mandatory STOP。**

直接基点 `62967269e611b61e73b70abe20e998d9c7eaaf50`、最終測定SOURCE `b7ad344401b10395c49e4cfa806909282859bab5`。branchは `track-b-g6-w2c-b3-real-api-adapter-contract-closed-v1-20261020`。内容固定commitと公開bytes照合はpublication_receipt.json、receipt-only最終HEAD自身の照合は最終chatの外部receiptに記録する。

Versioned static binding、F3/driver exact Codec sidecar、G7/F3-shaped Result/Failure、seed/Bits、Canonical固有Audit Seal/angle context、Full F3 angle/requery、人工backend、Global Bundle橋渡し、offline verifierを実装した。Canonical 3方式をFull F3 facadeへ変更していない。

- 新規unit/negative **102 PASS**、統合 **30 PASS**、semantic mutation **20/20拒否**。既存差分回帰 **107 PASS**は別集計。旧独立1,038は加算しない。
- 人工22 case：正常2、期待した技術失敗13、証拠不足7。未確認を正常へ補完しない。
- n256：16,896 planned/reserved/started/completed一致。1,536 full document＋15,360 hash-only document、全件worker/trace/state/receipt/callerへjoin。
- SOURCE89 pins、科学8 module/201 entry、旧291 SOURCE/test保全。実G7/F3 import/sampling・科学bits・新科学計算/数値再解析・実grant/claim・native・旧claim再利用・STOP解除は全て0。

[Q1〜Q10](handoff.md)、[最終検証](verification.json)、[case matrix](artificial_acceptance_matrix.json)、[未検証の5分類](remaining_real_science_validation.md)、[次の担当](pending_and_required_decisions.md)。同一作者の自己検証であり独立受入ではない。

## 再現

repository rootから以下を実行し、毎回未使用のoutput pathを指定する。実行するのは人工backendだけ。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 docs/research/track_b_g6_w2c_b3_real_api_adapter_contract_closed_v1_20261020/run_tests.py /tmp/adapter-units-NEW
PYTHONDONTWRITEBYTECODE=1 python3 docs/research/track_b_g6_w2c_b3_real_api_adapter_contract_closed_v1_20261020/run_matrix.py --new-output /tmp/adapter-matrix-NEW --jobs 3
PYTHONDONTWRITEBYTECODE=1 python3 docs/research/track_b_g6_w2c_b3_real_api_adapter_contract_closed_v1_20261020/run_integration_tests.py /tmp/adapter-matrix-NEW /tmp/adapter-integration-NEW 6
python3 docs/research/track_b_g6_w2c_b3_real_api_adapter_contract_closed_v1_20261020/restore_evidence.py /tmp/adapter-restored-NEW
```

最終実施ではIntegration/Mutationを独立partitionとして並列化した。全50 test IDとpartition実行scriptをarchiveに保存した。Rawと全member SHA256はevidence/archive_manifest.jsonを参照。offline trusted candidateはembedded候補を自動承認せず、保存時のcaller入力とSOURCE/runtime証拠へ照合する。異なるhostのruntime/path同値は未確立で、値やpathを黙って正規化しない。

次はローカルCodexの一括SOURCE独立受入と、ユーザー・GPTの正式科学契約/実接続scope判断。Codexは追加指示までSTOP。
