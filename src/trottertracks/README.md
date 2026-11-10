# 研究別SOURCE

- [df_threshold_reflection.py](df_threshold_reflection.py)：元Track A仮説Iのexact grid、可逆gate計画、
  暗黙辞書と既存RTE event bridge。[技術報告](../../docs/research/track_a_threshold_reflection_technical_20261110/REPORT.md)。
  人工整数／mock検査のみで、native builder・科学性能は未資格。既存科学SOURCEは不変更。
  [中心化・平方回避の追補](../../docs/research/track_a_threshold_centering_square_avoidance_20261110/REPORT.md)：
  明示的な共通scalar shiftと任意選択のcharge区間比較を追加。旧squareが既定、科学的総費用は未評価。

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

## 新A CT-0（2026-11-07）

[pr_reference_ct0.py](pr_reference_ct0.py)は固定Git blobのstrict record/key照合、位相域、RPE headroomだけを扱う。旧runner・合成・compileを実行しない。[REPORT](../../docs/research/pr_reference_ct0_20261107/REPORT.md)、[入力・結果・人工検査](../../artifacts/pr_reference_ct0/2026-11-07/README.md)。mandatory STOP済み。

## 新A CT-1（2026-11-07）

[pr_reference_ct1.py](pr_reference_ct1.py)：保存q1/r1のfinite strict synthesis/correctness。旧engine/emitter/builder/区間会計を再利用し、初回対象268 keyだけを扱う。[REPORT](../../docs/research/pr_reference_ct1_20261107/REPORT.md)、[入力・検査・STOP](../../artifacts/pr_reference_ct1/2026-11-07/README.md)。mandatory STOP済み。
