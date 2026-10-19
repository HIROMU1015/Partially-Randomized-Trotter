# 人工回帰

最終実行結果はconsumer_regressions/results_9_suites.json、integration_regressions/results.json、pilot_test_results.json。収集ID・実行数・fail/error/skip・SOURCE hash・ログを保存する。

|今回実行した集合|件数|意味|
|---|---:|---|
|consumer runnerの9 suites|540|既存6 suites275件＋serial85＋global77＋v3 consumer103。全PASS|
|既存v3 codec/F01/integration|125|30＋25＋70。上記540件と異なるtest ID。全PASS|
|新しいpilot候補|31|人工normal、race、spent、binding、STOP、unknown、F01、予算、receipt改変、費用。全PASS|

過去481件は今回のPASS数へ足さない。上記は収集ID集合でも照合する。全suiteで禁止された科学moduleのimportを検査した。新しい科学結果の生成はない。

途中経過をattempts/に保持した。導入初回は旧P1 fixture不足、その補充後にPython 3.12関数名assertion1件を修正した。新候補の保存付き試験ではfailure contextのFractionをJSONへ直接渡した5件が保存errorとなり、exactタグ保存へ修正した。その後statefile/物理bytes照合を追加した際、receipt改変testのJSON空白増加が先にbyte予算違反として検出されたため、同じ長さのcanonical保存に修正した。実装が通った過去試行を最終結果へ置換していない。

既存統合suiteの最初の起動はstdin経由でspawn子がmainを復元できなかった。通常の隔離プロセス起動へ直し同じ125件を再検証した。履歴はharness障害であり科学workerの再実行ではない。
