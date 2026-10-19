# Storage V2 S/P1 Production Bundle Bridge 技術候補

**人工技術候補を実装・検証した。科学入口を閉鎖し、公開照合後 mandatory STOP。正式採択・科学的 GO はしていない。**

公開測定 SOURCE は `e099dcc461247e81eff981867dc28c3f3b16df5e`。Producer `596781daeba253418ba35f4b7964312a54caf233` を直接基点とし、独立受入 `77126767c00585d0967558c0756907df2d69d1ff` を固定参照した。既存 branch の merge/rebase、旧 SOURCE/raw/claim の変更はない。

|最終集合|結果|
|---|---:|
|新規 bundle tests|107 PASS／FAIL・ERROR・SKIP 0|
|上記に含まれる mutation|35 拒否：integrity mismatch 32、証拠不完全 3|
|最終人工 matrix|43：bundle 正常 2、単独条件正常 2、技術失敗 13、証拠不完全 26|
|旧10suite 原版|622 PASS／1 FAIL|
|旧10suite versioned 最小補正版|623 PASS|
|既存 V2 storage／独立 consumer 回帰|41 PASS／57 PASS|
|重複しない補正版 regression ID|828 PASS（623＋41＋57＋107）|

43 matrix と35 mutation は107の内数。原版623件と補正版623件は重ねて加算しない。歴史108／680／820も加算しない。旧108の正常2／技術失敗40／証拠不完全66は固定参照のまま。

[Q1–Q10と次担当](handoff.md)、[architecture](bridge_architecture.md)、[技術契約](v2_technical_contract_candidate.md)、[差分](source_and_contract_delta.md)、[legacy保守](legacy_test_maintenance.md)、[global identity/予算](global_bundle_identity_and_budget.md)、[seed/F01](seed_prefix_and_condition_binding.md)、[Admission](admission_and_one_shot.md)、[STOP/storage](stop_unknown_and_storage_semantics.md)、[費用](cpu_wall_and_output_accounting.md)、[gate](production_gate_and_authority_boundaries.md)、[人工受入](artificial_acceptance_results.md)、[残件](pending_and_required_decisions.md)。

機械結果：[verification](verification.json)、[contract](technical_contract_candidate.json)、[static manifest](bundle_manifest_candidate.json)、[JSON schema](bundle.schema.json)、[matrix](artificial_acceptance_matrix.json)、[SOURCE inventory](source_inventory.json)、[固定公開照合](fixed_public_reference_verification.json)、[保全](preservation_verification.json)、[公開照合](publication_receipt.json)。

人工 raw は 7 part、archive 27,677,690 bytes／SHA256 `2d38d8b0a210b4428a7116a38bb44420c5a709a07861821c3b1444f44106a208`。全 2022 member を記録した。正常raw＋35差分overlayを復元して再読できる。実行済み保存先への上書き・同じfake scopeの再使用はしない。

```bash
python3 docs/research/track_b_g6_w2c_b3_v2_production_bundle_bridge_candidate_20261019/restore_evidence.py --new-directory /tmp/b3-bundle-new-restore
python3 docs/research/track_b_g6_w2c_b3_v2_production_bundle_bridge_candidate_20261019/verify_restored.py
python3 docs/research/track_b_g6_w2c_b3_v2_production_bundle_bridge_candidate_20261019/run_matrix.py reproduce --jobs 4
B3_BUNDLE_MATRIX=docs/research/track_b_g6_w2c_b3_v2_production_bundle_bridge_candidate_20261019/runs_reproduce python3 docs/research/track_b_g6_w2c_b3_v2_production_bundle_bridge_candidate_20261019/run_tests.py reproduce
```

既存回帰は `prepare_legacy.py` → `run_legacy.py reproduce`。固定の旧独立57件は77126767の指定packetを復元した専用参照treeが必要であり、`run_fixed_regressions.py` は使用した同一サーバー上の参照pathを明示している。新環境で公開分割rawを復元するだけの手順と、同じruntime bytesを要するテスト再実行は別である。Python 3.12／標準ライブラリのみ。SOURCE/input/runtime はhashで固定する。

作者はprimary Codex一名。offline判定をProducerから分離したが、別reviewerによる独立レビューではない。次の担当は **ユーザーとGPT**。
