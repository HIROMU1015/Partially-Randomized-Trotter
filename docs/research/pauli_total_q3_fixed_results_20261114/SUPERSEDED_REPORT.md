# 旧固定2 jobの保全停止

2026-11-14 JST。ユーザーが高速版への切替と、同じ固定条件での2件の再実行を明示許可したため、旧run01を途中成果のまま保全停止した。

自分のcollector、supervisor、LD1/LD2 workerのみを確認して停止。他担当のprocess・worktreeは変更していない。元SOURCE・input・PARTIAL_COARSE/LEMMA・進捗/資源log・旧DP結果は保全している。

[run01/SUPERSEDED_STOP.json](run01/SUPERSEDED_STOP.json)が停止状態の正本。旧SOURCEでは2科学callを開始したが、total DPは完了していない。この旧runを成功・十分N・最終MSE結果として扱わず、完了済みであったことにしない。旧runの新規COMPLETIONや最終モデルMSEを作っていない。

代替の実行は高速版の別namespaceへ保存し、同じLD1/LD2・同じ1vector・target/loss/IID・rho/finite scopeを各1回、計2件だけ実行する。変更点は明示された高速backend SOURCEと実行来歴であり、科学条件は変更しない。新しいshot探索、価格/G、主推定量採択は行わない。

この旧runは**mandatory STOP**。代替runも終了後STOP。
