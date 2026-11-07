# 研究別SOURCE

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

## Track C混合公式（2026-11-08）

[固定H4のBCH・全18round信号・paired価格検証](../../docs/research/track_c_mixed_formula_20261108/REPORT.md)。半刻み対照に対する資源改善は未観測、native全18round保証・絶対Gは未資格、mandatory STOP。新SOURCEは`track_c_mixed_*.py`、初期入口は`run_track_c_mixed_formula.py`、保存結果の読取検査は`check_track_c_mixed_formula.py`。
