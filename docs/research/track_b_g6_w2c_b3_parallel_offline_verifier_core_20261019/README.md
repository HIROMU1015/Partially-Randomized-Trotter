# 公開P1人工版の独立global verifier

2026-10-19 JST。**ARTIFICIAL_IMPLEMENTATION_COMPLETE / NOT_ADOPTED / mandatory STOP**。
既存360＋新77＝**437 tests PASS、FAIL/ERROR/SKIP各0**。既存SOURCEは変更せず、専用module・tests・Evidence Packetだけを追加した。

直接基点 `e4231b30919a4817f37d4b4d3c7a492a90fe4a7e`、branch `track-b-g6-w2c-b3-parallel-offline-verifier-core-20261019`。
公開prototype `1cd2a7196bc48a4aa1a904d7bd351e3d8fbf45ff` の実SOURCE/保存形式を基準とし、測定SOURCE `77f8667fc0898a3615a4efa834d971b3c339df71` は公開bundleを一時bare Gitへ復元してbytesを照合した。
親repositoryへそのbranchをmerge/rebaseしていない。サーバーの未公開統合版は取得・推測せず、別adapterは**PENDING_SERVER_SCHEMA**。

- [global/state/STOP検証契約](global_verification_contract.md)
- [公開人工入力の取得・検証範囲](published_evidence.md)／[18件の判定JSON](published_case_verification.json)
- [CPU/wall会計](parallel_cost_accounting.md)／[受入項目](integration_acceptance.md)
- [人工回帰](artificial_test_report.md)／[全test ID/log SHA](artificial_test_results.json)
- [Q1–Q9と最終schemaへの接続](handoff.md)
- [SOURCEと固定資料](source_inventory.json)／[入力の元bytes SHA](published_input_inventory.json)／[測定SOURCE復元](fixed_SOURCE_recovery.json)
- [保全](verification.json)／[公開照合](publication_receipt.json)
- [公開正常例](published_normal_example/verification_report.md)／[その費用](published_normal_example/cost_summary.json)
- [独立人工例](independent_artificial_example/verification_report.md)／[その受入matrix](independent_artificial_example/integration_acceptance_report.md)

新runtime moduleは `g6_w2c_b3_parallel_verifier_core.py`、`g6_w2c_b3_parallel_offline_adapter.py`、`g6_w2c_b3_parallel_offline_cost.py`、`g6_w2c_b3_parallel_offline_cli.py`。
いずれも `src/trottertracks/algorithm_codesign/` 以下。runtime verifierはproducer/scheduler/Admission/科学SOURCEをimportしない。
元pair/F01/cross verifierを再利用し、旧 `PendingParallelAdapter` は変更していない。

## CLI

repository rootで、入力とは別の未存在directoryを指定する。探索・overwrite・append・retry/resume機能はない。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /usr/bin/python3 -m trottertracks.algorithm_codesign.g6_w2c_b3_parallel_offline_cli \
 --artificial-manifest docs/research/track_b_g6_w2c_b3_parallel_offline_verifier_core_20261019/published_cases/light-after8-0/manifest.json \
 --new-output /tmp/b3-parallel-verification-new
```

verification/cost/acceptanceのJSON・MarkdownとCSVの計7出力。failureはTECHNICAL_FAILURE、破損はINTEGRITY_MISMATCH、欠測はINSUFFICIENT_EVIDENCE、未知schemaはUNSUPPORTED_SCHEMA。
正常完了という判定は、保存された技術証拠の整合性を示す。科学的正確性・物理RZ・実行authority・全操作履歴・物理的なprocess終了を独立証明したとはしない。
人工manifestのproducer_commitは**形式の固定参照commit**。独立fixtureはfixture_originで別記し、そのcommitがfixtureを生成したと主張しない。

8独立suitesの再現：

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 docs/research/track_b_g6_w2c_b3_parallel_offline_verifier_core_20261019/run_artificial_tests.py
```

logは新しいowned /tmpへ保存。新規SOURCEは6ファイル、基点の全既存tracked bytesを保全。研究上の主指標・科学validation pathを新採択しておらず、既存研究索引/status/manifestは保持した。
将来の対象契約はG7 P5_general_order、p=(1/5,3/10,1/2)、x5/7、m5、σ+1、4方式×A/B×X/Y、R1/A6 math-only τ10^-6/Kφ32/J32。
分子/geometry/basis/DF rank/L_D/δ窓は非該当。正式予定256＋15,855,296＋15,855,296＝31,710,848 callsは**静的照合だけ**。
今回は実G7/F3 import/sampling・新科学計算・実科学値再解析・実authorization/grant/claim発行消費・正式採択・科学レビューを行っていない。
