# 実G7/F3 API Binding候補：ローカル独立技術受入

判定：**閉鎖portの静的対応・人工保存証拠を限定受入。実科学接続とGlobal Bundle統合は未確立。Mandatory STOP。**

作業基点 `6bffb5b5819918698b522270ff19b255766d3072`。API候補公開HEAD `82896d53fa97235fa848d6c0833eb193af8b99d6`、人工測定SOURCE `78c571005fe766c61b32dc867e006531099ff608`。科学SOURCE `76ca3c4089b201d55da0fede08e1f77733ac38a5` はbytes/ASTだけを読む。これはローカル技術受入であり、科学的採択・本番実行許可ではない。

- [Q1–Q10と引渡し](handoff.md)
- [固定SOURCEと公開bytes](independent_source_verification.md)
- [科学APIの静的mapping](scientific_api_mapping_acceptance.md)
- [人工port契約](port_contract_acceptance.md)、[seed/bits](bitstream_and_seed_acceptance.md)
- [authority/preflight](authority_and_preflight_acceptance.md)、[閉鎖gate](closed_production_gate_acceptance.md)
- [monitoring/cost](monitoring_and_cost_acceptance.md)、[未接続境界](global_bundle_integration_boundary.md)
- [不一致・開発診断](producer_consumer_discrepancies.md)、[次の判断](pending_and_required_decisions.md)
- [人工回帰](artificial_regression_results.md)、[機械可読総括](verification.json)

独立既存845＋新規130＝**975 tests PASS**。Producer73は別集計。公開port2正常/2故障、port mutation22、固定Bundle43＋mutation35は別の証拠集合であり、test数へ再加算しない。

公開固定資料779件をGit/公開bytes/SHA256照合。API raw 12 parts、46,334,109 bytes、5,912 memberを全件検証。新規checkerはProducer port/exercise/authority/self-checkerをimportしない。

## 再現（人工証拠のみ）

repository rootから実行する。以下の `<packet>` はこのディレクトリ。出力は全て新規 `/tmp` pathを指定する。

```bash
/usr/bin/python3 <packet>/restore_artificial_raw.py --new-directory /tmp/api-acceptance-raw-NEW
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 <packet>/run_offline_acceptance.py --raw-root /tmp/api-acceptance-raw-NEW --new-output /tmp/api-acceptance-offline-NEW
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 <packet>/run_independent_tests.py --raw-root /tmp/api-acceptance-raw-NEW --new-output /tmp/api-acceptance-tests-NEW
```

閉鎖gate/fake authorityを再試験する場合は、`prepare_artificial_runtime.py --new-runtime /tmp/api-frozen-runtime-NEW` で固定62 SOURCE＋10 inputの疎runtimeを作り、`run_closed_controls.py --runtime /tmp/api-frozen-runtime-NEW --new-output /tmp/api-controls-NEW` を別processで実行する。これは通常の独立offline経路とは分離された人工Producer入口試験。実G7/F3はload前に遮断する。Producer73の再現commandと既存回帰は回帰資料参照。

次の担当：ユーザー/GPTが、この証拠を評価して別versionのGlobal Bundle統合作業を承認するか判断する。Codexは追加指示までSTOP。
