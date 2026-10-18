# 2026-10-18：PR A_COST並列起動・保存B/D診断

利用者が準備方針での実行と中断時の修正再実行を承認したため、[9scopeの実行](../track_a_pr_cost_batch_execution_20261018.md)を開始した。初期SOURCE `31ef62a12407010e9c1a07f9b37facedb8dff5fe`、fresh CPU使用に応じてAは6/6/10 worker、B/Dは各1 process。quota未公開のまま容量/inode確認・書込不能fail-closedを適用した。

H4/H6 DとB3件は完了し、H8 Dは保存mode名・文字列factorの結合に起因する再現不一致で停止。元六軸adapterと同じID正規化とmode明示対応をSOURCE `8ff002dd9524b49284fd8fb8e11354ee1ec18a00` へ固定し、人工7検査後に新attemptで正常完了（3800行/40比較、元PR88行再現、native exit0/reap）。旧STOPと消費は返却・改変せず保存。ほかの実行は専用SOURCE snapshotで継続し、成功済みstageは再実行していない。Bの無条件平均は未確定のまま。

次はCodexがAの完了/STOPとnative回収を確認する。今回のscope以外のsampling/fit/compile・科学判断へ自動進行しない。
