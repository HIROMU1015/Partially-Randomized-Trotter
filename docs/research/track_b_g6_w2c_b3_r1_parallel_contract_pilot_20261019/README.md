# B3 8 pair並列scheduler・R1契約候補

状態：**ARTIFICIAL_IMPLEMENTATION_AND_MEASUREMENT_COMPLETE / NOT_ADOPTED / RUN_READY_FALSE**。

公開統合baseline `564409487a35e15d73f1d0f0c2611824b400ab02` から独立worktree/branch `track-b-g6-w2c-b3-r1-parallel-contract-pilot-20261019` を作成。元driver/admission/host/readiness・科学SOURCE・正式R1は不変。追加moduleは人工port専用で、本番parallel入口を公開していない。

- 同一phase内の8 fresh pair process、X→Y共有、correctness→baseline→instrumentedのbarrierを実装。
- P1の協調STOP、P2の開始枠制限、global call/byte reservation、1 byte/trial状態file、全failure/prefix/未知結果を実装。
- **既存275＋新規36 = 311人工tests PASS**。72性能測定、17障害ケース、全runのS0とのevent/bit/正規化stream一致。
- P1・8 workerの元逐次S0比wall短縮はlight/mixed/heavyで **3.84/3.63/3.89倍**。総CPUは約 **62/56/31%増加**。並列化はCPU費用の無条件な改善ではない。
- 同期障害ではP1で開始済み8件のうち7件、P2 cap2で2件のうち1件が失敗通知後に完了。全費用・prefixを保存。
- 正式採択に必要な意味論の変更を**6項目**に整理。新科学計算・実G7/F3 import/sampling・実authorization/grant/claim発行消費・本番実行は全て0。

| 人工負荷 | 元逐次S0 wall/CPU秒 | P1 8 worker wall/CPU秒 |
|---|---:|---:|
| light | 10.3422 / 10.3538 | 2.6914 / 16.7601 |
| mixed | 5.8032 / 5.8166 | 1.5997 / 9.0969 |
| heavy | 5.7055 / 5.7202 | 1.4685 / 7.4692 |

いずれも同じ人工task集合・全3phase・3反復の中央値。実G7の性能予報ではない。S/P1/P2の管理・profiler/競合を含む測定条件を記録し、旧pilot倍率と区別する。

- [設計](parallel_scheduler_design.md)、[STOP/in-flight](parallel_stop_and_inflight_contract.md)、[one-shot/admission](one_shot_and_atomic_admission.md)
- [CPU/wall・公平性](parallel_cost_accounting.md)、[出力/失敗保全](output_and_failure_preservation.md)
- [全設定・ばらつき・障害実測](parallel_scaling_and_fault_results.md)、[CSV](measurements/scaling/summary.csv)
- [契約候補](r1_parallel_contract_candidate.md) / [JSON](r1_parallel_contract_candidate.json)、[Q1–Q10・次の担当](handoff.md)
- [最終SOURCE inventory](source_inventory.json)、[性能SOURCE inventory](measurement_source_inventory.json)、[保護SOURCE](protected_source_inventory.json)
- [311 tests](tests/summary.json)、[再読監査](artifact_verification.json)、[保全検証](verification.json)、[公開照合](publication_receipt.json)

性能SOURCE `b1dba4b6f282ffab413719aa000f18267caeb1d7`、障害/最終人工SOURCE `ef57714b35bbdd3912fadf13ee522e6a4ff491e6`。測定後の例外cleanup等の修正を性能測定へ遡及しない。元commitは [measurement_source_history.bundle](measurement_source_history.bundle) で保全。固定baseline取得後に `git bundle verify` / 専用checkoutへfetchして復元できる。最終公開commitのmetadataと測定commitは区別する。

共通AGENTS.md・worktree AGENTS.md・固定共通指示§8.4を確認。AI sub-agentなし、主agent自身の検査。共有venv/package/OS/cgroup/他ユーザーprocess・他worktreeを変更していない。結果公開・保全確認後mandatory STOP。契約採択・本番統合・ローカル最適化統合・実入力pilotへ自動移行しない。

完全content commit `06bfc3a905b31149aaec32a3d823bc8016d4ac09` の94ファイル・12,082,477 bytesをremoteから再取得し全SHA256照合済み。追加の [emit途中flush検証](tests/flush_emit_verification.json) もPASS（完全行1、論理完了0）。311件のsuite集計とは別の補足probe1件。receiptを含む最終commitは別途照合する。
