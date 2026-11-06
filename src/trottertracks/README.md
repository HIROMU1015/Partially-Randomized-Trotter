# 研究別SOURCE

[O2-S2 pair adapter](algorithm_codesign/s2_multi_event.py) は保存ordered-pair価格を既存Parent IS/統計へ接続する。
[途中結果・SOURCE・runner・test](../../docs/research/o2_s2_multi_event_validation_20261106/REPORT.md)：
六cell評価は有理数process転送でSTOP。元研究のSTOPは維持し、無断retryしない。

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
