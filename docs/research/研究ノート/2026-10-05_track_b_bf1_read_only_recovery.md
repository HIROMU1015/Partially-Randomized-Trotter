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
