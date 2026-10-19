# Storage V2 サーバー独立技術受入

**人工技術受入完了。証拠不完全66件を維持し、科学実行は閉鎖したままmandatory STOP。**

固定Producer公開HEADは `596781daeba253418ba35f4b7964312a54caf233`、人工測定SOURCEは `45bf46b4a833e71f138903fb511f2741e54043b7`。旧独立検証器 `49a0dcc9c8ec6dfc96eba5bb862cfdcb3d0fc866` を直接基点に、別セッション・専用branch/worktreeで実装した。同じサーバー上の検証であり、異なる物理環境での外部再現や第三者査読ではない。サブエージェントは使用していない。

|区分|独立結果|
|---|---:|
|公開人工raw|正常2／技術失敗40／証拠不完全66、合計108|
|物理stateとcounterのexact照合|82ケース|
|旧指定10suite・原文のまま|622 PASS／ラベル期待値不整合1 FAIL|
|その期待値だけを明示補正した別versioned harness|623 PASS|
|新規独立tests|57 PASS、うちnegative/mutation 51ケース|
|上記versioned回帰の重複しないtest ID|680 PASS、FAIL/ERROR/SKIP 0|
|別枠の新規外部writer故障|S/P1の2ケース、独立再読は両方INCOMPLETE_EVIDENCE|

Producer既済820件、公開raw108件、外部writer故障2件を680へ加算していない。元testの失敗を消したり原版623 PASSと表示したりしない。詳細は [回帰区分](artificial_regression_results.md) と [旧assertion最小再現](legacy_label_assertion_reproduction.json)。

[Q1–Q10／次担当](handoff.md)、[SOURCE/schema](source_and_schema_verification.md)、[予約会計](v2_reservation_accounting_acceptance.md)、[trace/STOP](v2_trace_and_stop_acceptance.md)、[物理証拠境界](v2_physical_evidence_boundaries.md)、[費用・予算](cost_and_budget_verification.md)、[不一致](producer_consumer_discrepancies.md)、[残る証拠・契約判断](pending_evidence.md)。

機械結果は [108件matrix](artificial_108_case_verification.json)、[各ケース詳細](acceptance_delivery/)、[新規fault/mutation](independent_fault_injection_results.json)、[修正前後比較](reservation_before_after_independent.json)、[最終tests](new_tests_delivery.json)、[検証総括](verification.json)、[保全](preservation_verification.json)、[公開照合](publication_receipt.json)。

独立consumerは `g6_w2c_b3_storage_independent_{identity,trace,adapter}_v2.py`。旧global core、pair/F01、CPU partitionを再利用し、V2固有部分をこのセッションで実装した。Producer/scheduler/Admission/Producer補助checkerを独立consumer processへimportしていない。Producer実行があるのは別processの人工writer故障と、指定済み人工回帰だけである。

## 再現

Python 3.12と標準ライブラリだけを用いる。fresh checkoutでは、必要なら同じrepositoryから指定のProducer公開commitと測定SOURCE commitを `git fetch --no-tags origin <完全SHA>` で取得する。既存出力への上書きや同じ人工storeの再実行は行わず、新しい作業領域を使用する。

```bash
python3 docs/research/track_b_g6_w2c_b3_v2_independent_acceptance_server_20261019/prepare_evidence.py
python3 docs/research/track_b_g6_w2c_b3_v2_independent_acceptance_server_20261019/run_acceptance.py reproduce
python3 docs/research/track_b_g6_w2c_b3_v2_independent_acceptance_server_20261019/run_new_tests.py reproduce
python3 docs/research/track_b_g6_w2c_b3_v2_independent_acceptance_server_20261019/run_legacy_regressions.py reproduce
```

新規faultの固定runtimeは `prepare_injection_runtime.py` で新設し、`run_independent_writer_fault.py S` と `... P1` をそれぞれ未存在の保存先へ実行する。`verify_additional_evidence.py` が別processで独立再読する。runtimeの119ファイルは測定SOURCEからそのまま取り出し、科学moduleはimportしない。人工Admissionが要求する `/tmp/b3-r1-artificial-*` 領域を守る。初回の名前不適合による拒否もpreflight logとして保持した。

新規51 negative raw＋2 writer rawは5分割partと [manifest](new_artificial_raw_manifest.json) に保存した。次のhelperは実行せず復元するだけで、273ファイルのbytes/SHA256往復一致を確認済み。

```bash
python3 docs/research/track_b_g6_w2c_b3_v2_independent_acceptance_server_20261019/restore_evidence.py --new-directory /tmp/b3-v2-independent-new-restore
```

Producer公開108 rawは元の固定公開archiveを参照する。`prepare_evidence.py` は公開part、archive SHA256、758 memberのbytes/SHA256を独立に検証して新設領域へ復元する。元archiveをこのpacketで再生成したものへ置換しない。

正式science SOURCE採択、production bridge、実G7/F3 sampling/import、新科学計算、科学保存値の数値再解析、実authorization/grant/claim、quantum native build/compile/synthesisはすべて0。pilot n=256・S/P1計16,896 callsは未採択候補、Full-N 31,710,848 callsも未実行。次の担当はユーザーとGPT。
