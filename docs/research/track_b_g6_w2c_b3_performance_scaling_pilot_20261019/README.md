# B3 並列化・高速化の人工性能検証

2026-10-19 JST。状態：**ARTIFICIAL_PERFORMANCE_COMPLETE_NOT_ADOPTED**。独立branch `track-b-g6-w2c-b3-performance-scaling-pilot-20261019`。

- 現行8 pair構造：8 workerでlight/mixed/heavyのwallを **6.79/6.02/3.95倍**短縮。
- 64人工task：64 workerで **36.78倍**、CPUは約11.643→12.645秒。これは8 pairの本番倍率ではない。
- heavyのGMP候補：8 worker wall **約1.34倍**、総CPU **約1.30倍**改善。mixedの利益は小さい。
- 全**172人工tests PASS**、全測定出力のchain/file readback、正規化したevent/trace/bit/bytes互換性確認。
- 並列化は通知後0新規開始でもin-flightが完了し、**現行逐次R1と同じSTOPではない**。buffer crashでは受理2行が残らない例を確認。

固定の研究対象は `P5_general_order`、p=(1/5,3/10,1/2)、x=5/7、m=5、σ=+1、4方式×A/B×X/Y。これらの実入力は実行していない。

実G7 sampling・量子回路build/compile/synthesis・科学保存値再解析・実authorization/grant/claim発行消費は全て0。既存科学SOURCE、契約、raw結果、ledger/marker、他worktreeを変更していない。GPUは強制せず、CPUを優先して実測した。

環境：AMD EPYC 7532、64 logical CPU（lscpu上SMTなし）、RAM約503 GiB、起動時available約469 GiB、Python 3.12.3、専用venvのgmpy2 2.3.2/orjson 3.13.0。既存quotaを変更しない。共有OS cacheはuncontrolled。

SOURCE：F01基点 `07f65831f045bf1fdbd4b370e44c7f59a9901ddd`。scaling SOURCE `734519386b38c999fa7f502d800e9e7e2df9402f`、その他suite SOURCE `2371261024176d9e343bc8c8109758d5571cbd59`。後者はnative encoderのDEL fallbackとbuffer crash probeを追加。scalingは旧revisionのreference経路だけを実行し、native encoderは使用していない。測定中のSOURCE書換えは行っていない。

- [測定方法・再現](methodology.md) / [SOURCE inventory](protected_source_inventory.json)
- [ボトルネック](bottleneck_analysis.md) / [scaling表](parallel_scaling_results.md)
- [高速化比較](optimization_results.md) / [条件付き外挿](full_workload_runtime_projection.md)
- [R1・F01・STOP](contract_compatibility_and_stop.md) / [Q1–Q8・優先順位・次の担当](handoff.md)
- [全172 testsログ](artificial_regression.log) / [機械可読CSV](measurements/scaling_summary.csv)
- measurements配下に全57 scaling/45 optimization/78 hot-path/14 STOP・crash測定の圧縮JSONL、環境、SOURCE hash、代表rawを公開。全rawは専用worktreeの `.pilot-runs/` に保持し、per-file SHA/bytesは各測定summaryに記録。代表raw以外の大量trial JSONLはGitへ重複格納していない。

正式採択・本番統合・実入力pilotは別途ユーザー判断。公開照合は [publication_receipt.json](publication_receipt.json)、保全は [verification.json](verification.json) を参照。確認後mandatory STOP。

追加の全trial監査：[all_trial_audit.json](all_trial_audit.json)。2,445ファイル・1,152,889完了人工trialのID/count/bit/chain/file SHAを照合し、重複0。replica間のglobal IDは `(run directory, task key, local trial_id)`。同じR1形のlocal IDを単独のglobal IDとは扱わない。

公開経路：Git CLIの認証がないため、接続済みGitHub Appで同一treeを公開する。公開commitのmetadata/SHAとローカル測定SOURCE commitを区別し、元の測定commitは [measurement_source_history.bundle](measurement_source_history.bundle) にGit objectのまま保全した。固定F01基点があれば `git bundle verify` とbundleからのfetchで復元できる。
