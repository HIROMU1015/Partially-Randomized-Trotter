# src/trottertracks/resource_applicability：資料とSOURCEの統合入口

2026-11-03。公開版の索引にあった資料リンクをまとめた。元の索引は全版を保存している。
当時の「次段階」「承認済み」「最新」は原snapshotの記録として読み、現在の実行指示へ移さない。

[main集約の範囲・版差の扱い](../../../docs/research/main_collection_versions_20261103.md) ／ [研究結果への統一索引](../../../docs/research/RESEARCH_RESULTS_INDEX.md)

## main内の資料・保存原版


## 元snapshot・外部資料

大きなデータ、原SOURCE指定、mainと異なる版は元の参照から辿る。

- [adoption review](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/76fec27f45b64879cdf5a86a32772ba78bd0a038/inventory/integration/p1_adoption_review_20261027)
- [old P1 implementation/test history](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/af54aeb7ebac042a3ca02379d61516ac635feef9/docs/integration/p1_df_pipeline_h4_20261027)
- [the fixed P2 handoff](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/76fec27f45b64879cdf5a86a32772ba78bd0a038/inventory/integration/p1_adoption_review_20261027/P2_HANDOFF.md)

## 元の索引

- [原版 7935831f63f6](../../../docs/archive/main_collection_versions_20261103/7935831f63f60663450cec7c168686d7d58f7bb9/src/trottertracks/resource_applicability/README.md.original) ／ [元snapshot](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af54aeb7ebac042a3ca02379d61516ac635feef9/src/trottertracks/resource_applicability/README.md)
- [原版 0045edf07570](../../../docs/archive/main_collection_versions_20261103/0045edf075702540776f15588af997b5767929b4/src/trottertracks/resource_applicability/README.md.original) ／ [元snapshot](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2c865c66171fc9d1972b67c1257c57e5684fe26c/src/trottertracks/resource_applicability/README.md)

## P1・研究C SOURCEの組（2026-11-03）

[選定したSOURCEと依存版・検査範囲](../../../docs/integration/main_p1_c_price_cohorts_20261103.md)を参照する。P1は人工・mock検査79件PASS、研究Cはpassive importと静的互換を確認した。
共有coreの変更前版は保存済み。過去の科学SOURCE・入力・結果・契約・STOPは元snapshotを使う。
