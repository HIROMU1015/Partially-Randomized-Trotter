# H4 geometry source固定・レビュー入口

`H4_GEOMETRY_SOURCE_FROZEN_AWAITING_REVIEW`。科学未実行、公開後STOP。
Branch `track-a-h4-geometry-source-20261006`。base `b662dbd72e49fa713a25c716f323843e547e973b`。
今回の認可はnew source port、pure/synthetic検査、二つのcommit、non-force pushだけ。

- [実装・検査範囲と未検証事項](../../../../docs/research/track_a_h4_geometry_server_native_source_implementation.md)。
- [D1〜D4実装条件の採用記録](source_adoption_v1.json)。過去v2記録は書き換えない。
- [source固定後blob/hash・closure・環境監査](source_freeze_v1.json)。actual SOURCE_COMMITはこの別資料にだけ記録。
- [次レビュー依頼](REVIEW_REQUEST_v1.md)：入力生成専用authorization作成へ進めるかの判断だけ。
- [最終合成結果](test-attempt-09.json)、[最終ログ](test-attempt-09.log)、[累積合成transpile監査](synthetic_transpile_reservations.jsonl)。
- [source段manifest](source_stage_manifest_v1.json)、[review段manifest](review_bundle_manifest_v1.json)。

最終94 tests pass、fail/error/skip0。synthetic transpileは失敗・再検査も含め25/64件。
初回74件error12、次回74件fail1/error1、追加89件fail1の開発失敗ログも保存する。
旧128 transpileは比較120（30 tasks×workers1/6/12/16）＋axis/phase4＋full-operator4で、今回再実行0。
分子アクセス/生成、実signal/sampling/build/compile、GPU、環境・他job変更、execution authorization発行、production runner launchは0。
旧247 source、保存6 JSON、準備25、v1 26、v2 30 bundleは不変。旧原稿・図・Track B・旧結果/status/validation manifestも不変。

実SCF、科学入力と物理operator、live process/cgroup/AS/RSS、production filesystem障害時の耐性、全campaignの資源内完了は未検証。
source合成検査はlocal evidenceであり、科学成立・immutable CI・独立外部再現ではない。
このbundleはexecution authorization/planを発行せず、generationやmapを起動しない。
