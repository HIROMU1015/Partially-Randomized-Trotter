# H4 geometry contract preparation v2

`H4_GEOMETRY_CONTRACT_V2_PREPARED_AWAITING_REVIEW_SCIENCE_NOT_AUTHORIZED`

2026-10-06 JST。base `7c1a3d43f61c5501a9e79206b7c60933f94b1077`からのレビュー採用案。
専用branch `track-a-h4-geometry-contract-v2-20261006`。今回の指示は契約作成・合成検査・commit/non-force pushだけ。
契約完成・科学条件最終承認ではなく、D1〜D4はレビュー待ち。source port/input generation/science/next stage認可false、未seal、mandatory STOP。

- [契約案](CONTRACT_DRAFT_v2.md)、[D1〜D4設定案](review_decisions_v2.json)、[レビュー依頼](REVIEW_REQUEST_v2.md)。
- [v1→v2差分](V1_TO_V2_DIFF.md)、[scope](scope_v2.json)、[zero-compute plan](zero_compute_plan_v2.json)。
- [二段階認可順序](stage_contract_v2.json)、[schema](plan_schema_v2.json)、[pure JSON validator](contract_validator_v2.py)。
- [新規合成tests source](run_contract_tests_v2.py)、[結果](contract_tests_result_v2.json)、[最終ログ](contract_tests_v2.log)。
- [静的source/environment監査](static_source_audit_v2.json)、[旧証拠照合](identity_preservation_audit_v2.json)、[最終監査](final_audit_v2.json)、[manifest](artifact_manifest_v2.json)。

新規検査320件pass、fail/skip0。旧129件のrunner実行0、旧記録は保存。
準備時のPython bytecode lookupをguardが読込前に拒否した[起動失敗ログ](contract_tests_v2_attempt01_startup.log)と、
固定wrapper semanticsに対するfixture期待値を直した[検査失敗ログ](contract_tests_v2_attempt02_fixture.log)も保持する。
最終suiteの科学import/protected access試行は0。最初の拒否はPython .pyc lookup1件で、科学cacheやデータは読み込んでいない。
科学入力アクセス/生成、signal、sampling、science build/compile/transpile、新synthetic transpile、GPU、環境/他job変更、authorization発行は全て0。

v1 bundle26ファイルはbyte-identical。v1 manifest33件は起点commit blobと照合する。
更新した索引8件はv2 manifestに収録し、旧v1 manifestを現在の索引へ合わせて書き換えない。
旧247 source、公開準備25ファイル、保存6 JSON、原稿・図・旧result/status/manifest・Track Bは不変。
過去synthetic128 transpileの内訳は比較120件（30 task×workers1/6/12/16）＋axis/phase4件＋full-operator4件。今回のtranspile0。

再検査は新しいprivate source-only directoryで行い、v1 validator/schema/artificial fixtureのpinned bytesも同じ相対配置に置く。
既存result/logは上書きしない。contract validatorをscience runnerとして使用しない。
最終suiteはlocal synthetic証拠でありimmutable CI・独立外部再現ではない。公開後レビューまでSTOP。
