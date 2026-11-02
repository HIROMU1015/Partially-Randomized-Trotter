# 改善P1＋Fused canonical＋R1/Admission/Host 技術統合

2026-10-19 JST。**ARTIFICIAL_COMPLETE / NOT_ADOPTED / RUN_READY_FALSE**。
独立branch `track-b-g6-w2c-b3-parallel-fused-r1-technical-integration-20261019`、基点1cd2a719。
最終回帰・測定SOURCE **`860fef4a3a8ebb5a46628f4e0364e5934b5af416`**。新v3 Host/Admission、spawn worker、cached observer、coalesced IPC、fusedを接続。
実科学入口はhost/直接helperとも非採択guardで閉じる。正式science releaseや実行可能な正式contract digestは作っていない。

**481人工tests PASS（既存保護356＋統合125、fail/error/skip0）**、**60性能測定＋12 warmup＋35障害観測**。
P→PFの総CPU削減はlight/mixed/heavy **5.6/4.0/1.8%**、wall削減 **4.0/3.9/0.7%**。
新Host外側STOPまでの5反復中央値で、heavyの追加利益は小さい。実G7の予測や正式estimand採択ではない。

- [統合設計](integration_design.md)、[SOURCE/manifest/承認境界](source_and_manifest_contract.md)
- [STOP・保存・費用](parallel_stop_output_and_cost.md)、[fused互換性](fused_canonical_integration.md)
- [481回帰・障害](artificial_regression_results.md)、[CPU/wall/RSS/I/O実測](artificial_performance_results.md)
- [Q1–Q9](handoff.md)、[ローカルadapter引継ぎ](offline_adapter_handoff.md)
- [技術契約案](technical_contract_candidate.json)、[schema/protocol](output_schema/protocol.json)、[runtime manifest](runtime_manifest_candidate.json)、[静的full schedule](full_schedule_static.json)
- [SOURCE inventory](source_inventory.json)、[元SOURCE履歴bundle](measurement_source_history.bundle)、[raw再読監査](artifact_verification.json)
- [保全・境界](verification.json)、[公開照合receipt](publication_receipt.json)

固定科学SOURCE・旧R1/Host/Admission・旧結果/契約/実ledger/marker・他worktreeを変更せず、新規pathだけを追加。
実G7/F3 import/sampling、新科学計算、実保存値再解析、実authorization/grant/claim、correctness256/full-N本番は全0。
共有設定や他ユーザーprocessを変更していない。検査はprimary agent自身による。公開照合後**mandatory STOP**。
次はローカルCodexのparallel adapter。正式採択/科学実行はユーザーの別判断・承認が必要。

公開照合：content commit `396b174cb492161b15f9dce23fc7ea3ce6911b56` の208ファイル・19,987,245 bytesを取得し直し、全SHA256とremote HEADを照合した。receipt自身を含む最終HEADと差分は別途照合する。
