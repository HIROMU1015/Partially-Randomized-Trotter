# mainへのSOURCE集約：P1と研究Cの版の選定

2026-11-03。前回保留したP1の6ファイルと研究Cの価格統計helperについて、
公開snapshotのまとまり、mainの依存API、元の入力・SOURCE対応を照合した。
各実装は元の公開blob・modeのまま配置し、元版のバイト列は保存している。

## P1

P1は `2c865c66171fc9d1972b67c1257c57e5684fe26c` に揃った6ファイルを配置した。
同snapshotの版は、明示的なstudy callback、準備policy、estimand境界を保ち、
mainと同じeager initializerを変更しない。元のP1公開版 `af54aeb7ebac042a3ca02379d61516ac635feef9`
が導入したlazy facadeは採用していない。

最初の人工・mock検査では78 PASS・1 FAILで、mainの `prepare_df_partial_s2` が
`partition_policy` を受け取れないことが分かった。この引数を削除すると固定順序の指定が失われるため、
P1文書に記録された公開core/base `f29eafbf4412193e8e8660d9f051889c2cba684b` の依存SOURCEを照合した。
依存関係内の31モジュールのうち、mainと異なる5ファイルをその公開baseと同一内容に揃えた。
これには固定順序のpolicy、optional basis planと対応builder、既存executorのSOURCEが含まれる。
変更前のmain版5ファイルも `.original` として保存した。

その後、既存のP1人工・mock検査 **79件すべてPASS**。
配線・拒否条件・未取得値・既存APIのsignature・SOURCE/cohort整合・新規出力の衝突拒否を確認した。
実H4の入力・演算子・信号・shots・回路は評価していない。各科学callbackはmockで、
人工fixtureは型・shape確認用のzero配列だけを用いる。
検査は既存Python 3.11環境を読み取りで使用し、専用一時出力・bytecode/cache抑制で実施した。
元のvenv、研究worktree、実行中jobは変更していない。

plan/syntheticの技術経路を確認した結果であり、real/scientific/resumeの実行機能を有効化していない。
共有coreの科学的等価性、全既存test、main全体の数値再現性を検証したとは扱わない。

### 配置したP1 SOURCEと依存版

| path | 原snapshot | blob |
|---|---|---|
| [scripts/run_df_research_pipeline.py](../../scripts/run_df_research_pipeline.py) | `2c865c66171f` | `3f276dc54960` |
| [src/trottertracks/resource_applicability/df_core_stages.py](../../src/trottertracks/resource_applicability/df_core_stages.py) | `2c865c66171f` | `56913258e496` |
| [src/trottertracks/resource_applicability/h4_geometry_pipeline.py](../../src/trottertracks/resource_applicability/h4_geometry_pipeline.py) | `2c865c66171f` | `880177473c83` |
| [src/trottertracks/resource_applicability/pipeline.py](../../src/trottertracks/resource_applicability/pipeline.py) | `2c865c66171f` | `86c59b5a86cd` |
| [src/trottertracks/resource_applicability/pipeline_fixture.py](../../src/trottertracks/resource_applicability/pipeline_fixture.py) | `2c865c66171f` | `66a4c8cd04e2` |
| [tests/test_df_research_pipeline.py](../../tests/test_df_research_pipeline.py) | `2c865c66171f` | `913279ab89a6` |
| [src/trotterlib/df_partial_s2.py](../../src/trotterlib/df_partial_s2.py) | `f29eafbf4412` | `f467ae0a9875` |
| [src/trotterlib/df_partial_s2_repeated.py](../../src/trotterlib/df_partial_s2_repeated.py) | `f29eafbf4412` | `6b88b58eff50` |
| [src/trotterlib/df_rte_circuit.py](../../src/trotterlib/df_rte_circuit.py) | `f29eafbf4412` | `82e125f3be62` |
| [src/trotterlib/df_rte_qiskit.py](../../src/trotterlib/df_rte_qiskit.py) | `f29eafbf4412` | `da3c7c68e6ee` |
| [src/trotterlib/parallel_validation_executor.py](../../src/trotterlib/parallel_validation_executor.py) | `f29eafbf4412` | `b0717707a880` |

元のP1実装報告・test結果は当時の公開版についての記録として保持する。
mainで選んだ版と今回の検査結果は本書を使い、過去の報告本文を新しい実行結果に書き換えない。

## 研究Cの価格統計helper

`src/trottertracks/c_r32_price_statistics.py` は、公開snapshot
`9837eac59b94113bf057fcc311a27b35e60343f5` の共通化版を配置した。
`c_update064_r32_price_statistics.py` が必要とするoptional `study` / `statistics` を持つ版である。
旧版とのAST比較では、top-levelの定数・importと、変更対象2関数以外の全関数は同じで、
従来のkeyword既定値も保持されていることを確認した。

| 系列 | SOURCEの扱い | 設計の区別 |
|---|---|---|
| update005 / A | 元helperは保存原版と固定SOURCEから辿る。mainの共通化版に旧SOURCEの認定を移さない | 元の640件・0.5%半幅目標の既定値を保持 |
| update064 / B | 共通化版はBの元module SOURCE hashと一致。studyとstatisticsを明示する呼出に対応 | Bの256件・1%半幅目標を別指定し、Aと混ぜない |

数値ライブラリのimportを拒否する別processで、両moduleのpassive import、共有helperの参照、
optional引数と別々の設計定数を確認した。統計関数・execute・nativeは呼んでいない。
Aの元moduleは保存原版のSHA256がAの既存SOURCE記録と一致し、mainの共通化版はBの既存記録と一致する。
これはmodule単体の対応確認であり、各SOURCE manifestに含まれる全ファイル・runtimeの資格確認ではない。

旧Aの計算結果をmainのhelperで再解析しない。元の科学SOURCE・入力・契約・cohort・seed・status・STOPは保持する。
既存のSOURCE manifestや契約のhashをmainへ合わせて更新していない。
mainからの科学実行を新たに許可・保証するものではなく、各実験は元の固定snapshotと契約に従う。

## 保存と確認範囲

[版と出典の対応表](../research/main_collection_versions_20261103.tsv)は、元の37 path・88版に共有core 5 pathの旧／公開版を追加し、
現在は42 path・98版を記録する。原版保存と実行pathへの配置を区別している。
研究ノート4・発表資料4・過去の承認記録2は、各原版を保存したまま単一版を選んでいない。

新しい科学計算、保存結果の数値再解析、回路build/compileは0。
研究Cの数値test・科学runner、共有coreの科学検証は実行していない。
既存のraw、入力、科学SOURCE参照、契約・grant・ledger・marker・STOPと保存タグを保全する。
枝全体のGit mergeとブランチ削除は行っていない。

サーバー1・2は既に渡した固定main `fe2938191f7821b28e565db2880b2d66fbec5f46` を比較基準としてよい。
結果を統合する際に、本書の後続差分をローカル側で照合する。
