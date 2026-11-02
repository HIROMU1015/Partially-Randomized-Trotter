# 人工回帰の区分

最終独立回帰は **623 PASS=既存540＋新83、FAIL/ERROR/SKIP各0**。10 suiteを分離processで並列実行し、collected ID623個の一意性、SOURCE SHA、gzip log SHA、禁止科学module import0を記録した。artificial_test_results.jsonが原本。ローカル実行でありCI/外部再現と呼ばない。

新83件は公開18case、state/F01/physical prefix/call/byte/seq/chain/canonical/phase/STOP/drain、candidate/source/runtime/schema/scope/claim/offset、receipt/tail/ACK/unknown/CPU、提案N/modes/seed改変、read-only/no-overwrite、production guard fragment、bounded trace memory、新S projection、producer trace-write故障の検出を含む。

別に固定producerの31人工testをSOURCE bytesそのままでowned runtimeで再実行し全PASS。新しいfake18 artifactsも保存してSQLite snapshotを独立再読した。この31件を独立623件へ足していない。producer報告696/481や既存統合125件を今回のlocal PASS数へ加えていない。

独立collector初回77 PASS、その後guard3・fresh replay2・truncated gzip1の6件を加えた最終83 PASS。producer側で追加故障を発見したことと、検出器のtestがPASSしたことを区別する。SRC_TRACE_RESERVE_ACCOUNTINGは未修正の技術不整合であり、PASS623を理由に受入済みへ変更しない。

bounded-memoryは1,000→40,000 trace行でtracemalloc peakが一定範囲にあることを実測した。数値はpilot_bounded_memory.json。旧pair/state/archive streaming testsも再実行済み。3個のsynthetic log workloadは別の負荷artifactで、unit test件数へ加えていない。

実G7/F3 SOURCE import/sampling、新科学計算、保存科学値再解析、実authorization/grant/claimは0。人工replayに用いたprivate SQLiteだけを新しく作り、共有package/cache/OS設定/他jobへ変更していない。

最終費用異常検査の追加時に、同じreport参照へ前の変造値を残したfixtureで1 ERRORが出た。独立deepcopyへ修正した。費用不整合は全体INTEGRITY_MISMATCH・受入FAIL・比較CSV除外へ伝播する。開発試行log/sourceはlogs/cost_fixture_attempt*、最終結果と区別する。
