# 2026-10-11 N3保証生成・N1decode復旧の準備

利用者が完了済み科学レビューと次段方針を提示した。科学batchは未承認と明記しているため、
今回は[入口](../n3_generator_n1_recovery_preparation_20261011.md)と二つの結果前契約を作った。
N3はμの下限を代入した一変数LMI、候補数値探索と有理PSDの分離、同じU/cの比較、
12有理係数入力の移送scopeを準備。N1は実形式decoder境界・全context preflight・旧条件一回N1専用scope。
N2は保存結果のまま。既存のsource/失敗・N3/N2結果を変更しない。

これらは設計上の導出とmetadata準備であり、数学の実行検証・solver・対角化・LP/compile・科学testsは0。
decoder修正済みや算法実装済みとは表現しない。旧GPT付録を再実行しない。
resource read-only inventoryを記録し、将来CPU最大2 workers/各thread1、memory/output/call caps、wall capなしを契約化。
新しい科学実行には各作業への明示指示とsource/plan/authorizationの固定が必要。
同じ科学結果の再レビューはせず、新結果取得後の科学判断に戻す。Mandatory STOP、中心テーマnull。

旧manifest88 entriesはそのまま保存し、新準備entryだけを追加する。既存global checkerのdescription欠落は未修正。
root dirty状態・他Trackを保つため公開10d91e51基点の別worktree/branchで準備した。
