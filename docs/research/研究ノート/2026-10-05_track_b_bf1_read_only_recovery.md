# 2026-10-05 Track B: BF1-R0 authorization and preparation

利用者からGPT review `PROCEED_READ_ONLY_BF1_RESULT_RECOVERY_BEFORE_REDESIGN`を受領した。
原BF-1の科学再実行ではなく、保存済み1269 cellと元source/domainだけによる
deterministic replayを一回認可する指示である。再実行例外は認めていない。

別branch `track-b-bf1-read-only-recovery`をserialization修正commit `abd2425`から作り、
[R0契約](../../tracks/algorithm_codesign/bf1_read_only_recovery_contract_v1.md)、B専用module/runner/testsを準備する。
原sourceの探索・objective・primary規則を再利用し、physical initializationとsignal取得を拒否する。
cache欠落でSTOPし、bridge欠落はprimary/attributionと分離する。原resultとscience markerは保持する。

contractとsourceをcommit固定してから原cacheのreplayを一度行う。全outcome後にmandatory STOPし、
新artifactとレビュー資料を公開して研究方針判断をGPTへ戻す。以下の準備記録は結果前の履歴である。

同日、contract/sourceを`f226f8c81b5ba07b6d0c4b248c0eb15bf080a622`へcommit・pushし、
限定23 testsと全text identityを固定した後、一回だけread-only replayを行った。
`BF1_READ_ONLY_RECOVERY_COMPLETE`。事後のpreregistered primary復元はBF-A、primary ratio
1.6024124982、F/Lのfinite最小値の比は1.000で、双方Suzuki5のq1/R10/K2だった。
共通参照の最小値はnative S2 q2/R5/K2。bridgeのK4 cell欠落は穴埋めせず保存した。

[結果・GPT handoff](../../tracks/algorithm_codesign/bf1_read_only_recovery_result_validation_20261005.md)へ記録する。
原resultはINCONCLUSIVEのまま、原cell・preparation・science markerも不変。
R0 markerもconsumedでretryなし。新cell/signal/追加science candidate、science rerunは0。
mandatory STOPし、研究方針・RQ・新規性・論文着地点の全面再評価はGPT側へ戻す。
