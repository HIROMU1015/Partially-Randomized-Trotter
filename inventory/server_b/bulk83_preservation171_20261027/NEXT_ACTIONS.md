# 次の判断と担当

| 担当 | 判断・作業 | 今回との境界 |
| --- | --- | --- |
| ユーザー | READY83_IDS.csvの正確な83 IDに対する実削除をまとめて指示するか判断 | 利用終了判断は確認済み。今回の実削除許可は0 |
| Codex（その明示指示後） | 直前preflight後、BULK83_EXECUTION_PLAN.mdに従って合格IDを通常removeし、保持資産を監査 | 今回remove・branch削除・mergeは実行しない |
| ユーザー/研究担当 | NEW_TECHNICALLY_READY_IDS.csvの86件について今後の利用・再開予定の終了を判断 | 元171件に83件の利用終了判断を転用しない |
| Codex（後続指示後） | 新候補の残るローカル変更・ignored資産の扱いを含む具体的な実削除方法を準備 | 保全後も原worktreeはdirty/ignoredを含み得る。通常removeが今すぐ通る認定ではなく、force/clean等の許可もない |
| ユーザー/該当研究担当 | DEPENDENCY_HOLDS.csvの固定SOURCE/control/runtimeの役割と物理path保持要否を個別条件から判断 | 保全完了と依存解除は別。旧契約やSTOPを変更していない |
| 該当環境/研究担当 | 旧63件の未解消条件を既存ID対応で回答 | 今回全件再走査せず、他環境へ無断接続しない |

現在登録379件。指定範囲の技術作業と匿名化公開の確認後mandatory STOP。B3や科学実行への自動移行、実削除、branch操作は行わない。
