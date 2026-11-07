# C-S V0 schema修復SOURCE

2026-11-08 JST。旧結果64fcb50のrun01/TypeError/STOPを保全したまま、今回の明示許可に従い技術修復・再実行を進める。科学条件は採択済み値を維持する。

保存basis parameterはdf_rte_tail.describe_basis_change_operationsがstr(parameter)として保存し、既存a1_engine_connection.restore_operationsはfloatへ戻している。今回、inverse boundaryの符号反転にも同じ復元を入れた。保存値を別精度へ近似する変更ではなく、既にforward boundaryで用いていたcodecとの整合である。

新規人工5件が初回PASS（0.083秒）：str/数値入力の同一binding、XXPlusYY thetaのみ反転/beta保持、unsupported str STOP、新outputと旧STOP保全/二重実行拒否、完全な新SOURCE lockと改変STOP。旧18/9/10件の全面再実行はしない。新native provider・law・kernel・価格器は作らない。

旧source/input/runtimeの保護は有効。変更core/preflightのhashは新SOURCEの完全表に登録し、旧科学SOURCE.jsonや旧STOPを上書きしない。新outputはcampaign_run02、新cache。baseは64fcb50、branchはcodex/cs-v0-repair-20261108、独立worktreeを使用。

このcommitは科学入力投入前の修復SOURCE。公開SHA/body/blob/treeを照合してから、今回許可された同一条件の限定再実行を行う。実結果は別の結果commitへ保存する。科学的cap不成立は救済せず、完了/科学的STOP後はmandatory STOP。旧main/dirty/untracked、固定13入力、旧SOURCE/result/STOP、C1凍結、他jobと共有環境を保全する。
