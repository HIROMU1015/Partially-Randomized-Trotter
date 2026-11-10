# 研究別SOURCE

[R価格Phase 1実行結果](../../docs/research/r_price_phase1_results_20261111/REPORT.md)（2026-11-11）：Linear H4/H6・1Å・STO-3G・保存DF10/19、H4 S2/δ.1/p1/K2とH6 m5_best/δ1.6/p7/K2。SOURCE1a87e5331で通常64＋別rare16 compile、保存C/S59 path再利用。係数を7eeabf0aで検証取得前に凍結し再fitなし。独立n8の相対RMSEはH4 q16/R8で1.039841%、H6 q2/R32で0.229959%。有限cohort観測、採用閾値未設定・model未採用、全law/長q-R/総CI/全Gは未資格。新signal/N/G0、公開照合後mandatory STOP。以下の準備記録・過去STOPは履歴として保持。

[R価格Phase 1の実行準備](../../docs/research/r_price_phase1_preparation_20261111/REPORT.md)（2026-11-11）：保存C/S接続・paired3係数fit・Git凍結後validation・別unit rare診断の専用runnerを準備。公開A/B metadata整合、新14＋旧R回帰9＝23人工/mock PASS。固定SOURCE/PLAN、最大64＋16 wrapperの科学scopeは未実行。実分子sampling/S再計数/compile/fit/価格予測/再解析/N/G0、公開照合後STOP。

[R可変compiled価格の調査・設計](../../docs/research/r_variable_compiled_price_design_20261111/REPORT.md)（2026-11-11）：H4 S2/δ.1/p1/K2・H6 m5_best/δ1.6/p7/K2の固定Rモデルを保全し、構造価格＋少数compiled残差補正を提案。R binding/保存lookup adapter、人工/mock9件PASS。新R Cはnull、実科学計算・価格取得/予測/fit/G0。最小新q×R/希少event計画は未承認・未実行、公開照合後STOP。

[H6 provider統合・H4/H6全13round N/C coverage](../../docs/research/h6_provider_integration_nc_coverage_20261111/REPORT.md)（2026-11-11）：固定B cb2 APIへ保存lookup factoryを追加。C provider/テストをbyte一致再利用、24技術test PASS。H4公開89c91c69の整数検証済みNとH6公開f4c0bdaのNを静的転記し52行条件付きPASS。直接CはH4 5候補/H6 8候補、PR全law/長q/共有CI/全RPE採用は未資格。新科学計算・実N×C/G0、B実行領域へ非介入、公開照合後STOP。

[H4 price providerのB接続](../../docs/research/h4_price_pipeline_integration_20261110/REPORT.md)：h4_df10_price_providerと固定V1/Mq依存を取込み。B factory/run_missing/coverageへ接続し、34技術test PASS、実N/C/G0。

共通 `trotterlib` APIと区別した実験別SOURCEの名前空間。
[mainへの収録範囲・原SOURCE・元の制約](../../docs/research/main_collection_20261103.md)を参照する。

- [G6 return aggregation prototype](algorithm_codesign/return_aggregation.py)：元snapshotの内容を保持。native資源評価は未取得、mandatory STOP。

## 他系列のSOURCEと版差

[研究別SOURCEの公開版と配置](../../docs/research/main_collection_versions_20261103.md)を参照する。P1 pipelineとprice-statisticsの旧版は保存原版から元snapshotへ戻る。選定したmain版は下記のSOURCEの組を参照する。
同名SOURCEの全版を保存し、元の入力・契約・STOPを区別している。

## P1・研究C SOURCEの組（2026-11-03）

[選定したSOURCEと依存版・検査範囲](../../docs/integration/main_p1_c_price_cohorts_20261103.md)を参照する。P1は人工・mock検査79件PASS、研究Cはpassive importと静的互換を確認した。
共有coreの変更前版は保存済み。過去の科学SOURCE・入力・結果・契約・STOPは元snapshotを使う。

## サーバー照合からの成果収録（2026-11-03）

[収録ファイル・保存版と確認範囲](../../docs/research/main_server_results_collection_20261103.md)から、過去の公開成果と元SOURCEを辿る。
既存mainの科学ファイルは上書きせず、研究test関連の原版は実行contextと分けて保存する。新科学計算は行っていない。
