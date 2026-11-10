# 研究別SOURCE

[R screeningと価格Phase1整合](../../docs/research/r_screening_price_alignment_20261111/REPORT.md)：元p/r/b provider実装を変更せず、metadata CLIで候補/SOURCE/law差と実験依存関係を照合。実科学計算0。

[R可変N/C provider統合](../../docs/research/rpe_provider_integration_20261111/REPORT.md)：pr_r_dependent_n/h6_df19_price_provider/rpe_nc_coverageとrpe_provider_integrationを追加。既存p/r APIを再利用し、新R欠測・N比較policy・価格資格を保持。83技術test、新実N/C/G0。

[R可変PF/PR費用](../../docs/research/rpe_r_selection_20261111/REPORT.md)：rpe_r_selection.pyとcheck_rpe_r_selection.py。R-only menuとA/C law/domain形式、exact積和・条件付き選択、39技術test PASS。実科学計算/実R最適化0。

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
