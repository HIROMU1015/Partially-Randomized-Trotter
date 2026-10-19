# 構造検証記録

2026-10-19。Codex主担当が実施。別担当の独立科学レビューではない。

`python3 scripts/integrate_h468_paper_evidence.py --check` がPASS。175保護ファイル、Figure 2原本34ファイル、6主表行、manifest1576行、CEM14行、ledger316行、保存値の出典pointer/CSV record、旧行prefix、H8重複alias・公開監査不足を検査した。

`python3 -m unittest discover -s tests -p test_h468_paper_evidence_structure.py -v`：11件PASS。保存SE改変、cohort混合、H4の偽再現、H8重複、公開監査不足の消去、新H4比の捏造、未観測分散0埋め、旧因子転用、最終freeze、原図改変を使い捨てmetadata上で拒否した。成功ケースも含む。科学モジュールはimportせず、raw trajectory費用は読み込まない。

GitHub receiptは実GETした19commit/47資料、SOURCE87hash照合を記録。H6大規模manifestのリモート取得は先頭2行とblob identityまで、全行は固定Git objectから読み込んだ。H8詳細監査は非公開のため未検証として維持する。

新規科学call・統計解析・sampling・compile・raw再集計0。共有設定変更・他job操作・AI delegation0。新規統合資料の専用branchへのpushは利用者の回答で承認済み。最終freeze/詳細科学レビューには進まずmandatory STOP。
