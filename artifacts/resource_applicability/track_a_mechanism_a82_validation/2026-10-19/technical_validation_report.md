# 技術確認結果

A82の実科学one-shotは1回のみ。固定SOURCE5 file、入力9件、runtime68 fileを実行前後に照合。新receipt/grant・marker/ledger/claim・output SHA/schemaを確認し、親/workerともnative exit0、wait/reapを確認した。旧保全対象312件は実行前・実行後・統合後もSHA一致。旧契約やSTOPを変更していない。

今回に直接関連するA82とH4 P/図の人工回帰27件PASS、失敗/skip0。すべてinvented/mockデータを使用し、有効production grantや保存科学入力をテストへ渡していない。前段158件は今回再実行しておらず、この27件と合算しない。

統合のbyte/metadata照合33群PASS（[チェック一覧](integration_checks.json)）。旧32行主表・64軸core・62軸比較因子・32行表示計画・14行H4 P別表と、図9 file/116点は不変。新side overlay64行・比較overlay32行が旧ID/cohort/n/statusと対応し、A82.PRのみ新規検証に分類される。旧8 Claim–Evidence行は全文保全。過去のmachine manifest全entryとindexの既存本文も保持した。

PRのnormalization・signed bias/headroom・log bound・保存ceil整合は固定runnerの成功結果を転記し、統合checkでは科学式を再実行していない。64 side検証と31比較因子の違いを確認。PFはexact保存copy、core/他63 side/新PR-PF因子/統計/sampling/compile/rankingは0。

初回の統合checkerはrun出力を9件と誤指定して失敗した。固定契約の出力5件＋marker/ledger/claimの3件＝8件に修正した。次の確認では、この最終報告を生成する前にREADMEからの先行リンクを検査したため停止し、生成順序に対応した確認へ修正した。両checkerと失敗記録を保全。全出力SHAは初回から一致し、科学的な不一致や追加実行は0。修正は統合checkerだけで固定SOURCE・契約・科学結果は変更していない。

これらは主担当自身のlocal technical checks。独立review・immutable CI・詳細科学レビューではない。新資料からprivate絶対path、raw signal/bias、seed/control情報の混入がないことを確認した。公開承認は未確定でpushなし。最終evidence freeze・他job/shared設定変更なし。
