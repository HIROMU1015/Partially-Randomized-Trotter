# 旧co_qualname testのversioned保守

原版は49a0dcc9のtest bytesを専用legacy_runtimeへ復元し、そのまま実行した。結果は **622 PASS／1 FAIL**。対象は `CandidateAcceptance.test_cached_observer_call_count_label_exclusive_partition`。実labelは `...<locals>.leaf`、旧期待値は `endswith(':leaf')`。

`versioned_legacy_test_v1.py` は原fileを変更せず別moduleとしてcompileする。文字列期待値1箇所だけを `endswith('.<locals>.leaf')` へ置換し、元filenameをobserver filterに使う。Scoped/Cached全測定行の等値比較、call数、exclusive partitionとresidualのassertionは原文のまま。補正版は **623 PASS**。

原版 SHA256 `0d5b0f541c971b12f5c7effc26922da5f536981f63d9771cc85d857b9e4f3914`、補正後compile対象SHA256 `e8654c4c2b57f5fecccb55d2d7fec4fba4c7a071efe7c6365b74354c3e2ab764`。[機械差分](legacy_test_delta.json)、[1行diff](legacy_test_minimal.diff)、[全旧回帰](legacy_regression_results.json)、regression_logsに失敗traceと補正後記録を保存した。原版を623 PASSへ記録し直していない。CPU測定定義や科学契約は変更0。

最初の限定復元では136依存不足ERRORがあった。固定artifact/fixtureのGit bytesを補完した後の最終値が上記。初期記録はdevelopment_diagnosticsに保持し、最終件数へ加算しない。
