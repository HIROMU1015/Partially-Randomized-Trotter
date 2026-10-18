# 人工回帰テスト

**既存64＋追加41＝105件PASS。実科学入力による数値再解析0。** 主担当による技術検査であり、独立レビューやimmutable CIではない。

実行：`python3 -I -B tests/test_mechanism_analysis.py`、`python3 -I -B tests/test_mechanism_explanation.py`。独立した2技術processで実行し、privateにstdout/stderr全文を保存した。科学runtime・Qiskit・NPZ・実分子fixtureは使わない。

|確認|人工結果|
|---|---|
|旧不具合の再現|人工step b／累積log_Bで旧guardがNORMALIZATION_LOG_BINDING、人工PF margin省略でLOG_BOUND_MISMATCHを再現。新typed入力の同じ人工例は説明COMPLETE|
|意味・表現|stepと累積、nominalとempirical upper、保存upper binary64とexp経路を分離。誤ったstage、負margin、nonfinite、PFの非zero nominal log、誤routeを拒否|
|数値規則|headroom・log bound・整数整合性、LOG_DOMAIN_ONLYとoverflow、underflowを0へしないこと|
|schema・metadata|単一signal／signals配列／listを処理、未知container拒否。candidate／cohort／binding／axis不一致を拒否|
|core独立性|旧core計算関数を呼ぶと失敗するmockで説明専用処理を確認。保存値・SE・shotの不変、core未完了や誤IDの拒否|
|認可・I/O|未承認grantは入力前に拒否。旧scopeの再利用拒否、旧STOP配下の書込拒否。人工CLIの実child exit0・結果hash・coreコピーを確認|
|親受入|coreコピー改変、数値UNRESOLVED、未登録sideの欠測を拒否|

初回の追加32件もPASS。その後、metadata転記で単一signal containerが未対応と判明して準備attempt01を保持し、対応テストを追加した。attempt02はPF overlayの未保存RTE専用b/Rを必要fieldと誤分類したため、実データの値を埋めず型を修正し、人工検査を追加。attempt03は63 sideに必要field、A82_OLD_PR.PRのみ未回収となった。これらはmetadata/schema準備の履歴であり、科学one-shotの再試行ではない。

原core・元結果・消費済み認可記録等75fileのSHAは初回保全receiptと一致。新24入力roleもSHA照合済み。人工PASSとfield存在を、実科学データの説明PASSに読み替えない。
