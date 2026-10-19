# 文書・構造検証

2026-10-19。Codex主担当の技術チェック。別担当の独立科学レビューではない。

- `python3 scripts/check_h468_paper_display.py --receipt`：PASS。197保護ファイル、原図34ファイル、byte同一の主表CSV、6条件とCEM/台帳の対応、A82履歴/後続完了、solver/正規化例外、H8 PF公開receiptとPR未公開監査の区分を検査。
- `python3 -m unittest discover -s tests -p test_h468_paper_display_structure.py -v`：13件PASS。表費用改変、原図変更、cohort relabel、H8二重計数、未承認数値処理、入力完全同一性、母平均保証、stale A82、新P因子転用、最終freezeを拒否。H8監査statusとCEM参照の改変は、新しい文書hashを付けても拒否する。
- `python3 -m unittest discover -s tests -p test_h468_paper_evidence_structure.py -v`：既存11件PASS。旧統合の保存値・cohort・監査範囲の回帰を確認。

GitHubから実取得した17公開資料のblob identityを固定Git objectと照合した。新しい科学module import、科学計算、統計、raw再集計、sampling、compileは0。旧SOURCE/結果/grant/ledger/marker/STOPと原図を変更していない。原private witnessを開いたり公開したりしていない。

表示固定は科学的最終採用・COMMON_CORE_READY・最終evidence freezeではない。承認範囲の新文書・構造コードだけを専用branchへ公開し、remote SHA確認後mandatory STOP。
