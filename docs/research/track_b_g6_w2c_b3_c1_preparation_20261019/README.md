# G6/G7 W2c-B3/C1：実費比較の技術準備

2026-10-19 JST（依頼系列は2026-10-18から継承）。状態：**`PREPARATION_COMPLETE_WITH_ADOPTION_AND_PROVIDER_GAPS`**。補助SOURCE・schema・guard・人工テストと未採択契約候補を準備した。**実G7実行・実費測定・native build/compile/synthesisは0件**。完全な実実行runnerの完成・実費優位・新科学条件の採択を意味しない。

## 最初に判断可能範囲を分ける

| 費用座標 | 既存証拠で分かること | 新たに必要なもの |
|---|---|---|
| B1/B2：N、digital acceptance、logical/helper係数、有限RZ価格 | 登録済みG7 P5ケースの保存値と限定的な期待T affine優位を再利用できる | 同じN/受理率を計算し直す必要なし。新価格座標への縮約は別途承認 |
| 全trial古典CPU | F2上界・旧128-trial診断はあるが、Fullの実全費用優位を認定できない | 比較schedule・費用estimandを採択し、B3実測。短trace→母期待費用の格上げは禁止 |
| F3監査込み比較 | Full既存F3は保持。canonical用の有限law witness/共通監査adapterを新設 | 新adapterの実G7接続は未実行。angle policy・実run driver固定後に初回correctnessと計測が必要 |
| native CQ/T/CX/depth | 旧有限RZ cacheはあるがCQは抽象、helper CXはskeleton-only | 実Q/encoding/control/ancilla、gate set/compiler/最適化、誤差互換性を選定・採択後にC1取得 |
| prep/readout/backend込み総費用 | 旧affine記号とE2E不足を保持 | C2が必要。今回は実装・測定対象外、未知価格はnullのまま |

対象は既存`P5_general_order`、p=(1/5,3/10,1/2)、x=5/7、m=5、4方式。これは**G7のP5入力**であり、G9のP5Closed法ではない。新しいHamiltonian/入力/次数・baseline探索はない。分子/geometry/basis/DF rank/split/δ窓の評価は該当しない。

固定source `ab2549f41b3546fb3940342a2162dd9ee93699c4`、K `28cfabb1d47e0e1824bce1e154a9c289738fa2b9`、F3 `1aea9b8c228c5b940f2fcba0f94493b3ffdec46e`、直接の基点[W2c-A][A]を維持する。G7のp順序、law、H160/K256、η/ρ、root/ceil、primitive error、固定confidence/biasは変更しない。保存値の転記先は[既存証拠再利用表](existing_evidence_reuse.json)。引用であり、再計算・再ランキングではない。

## 成果物

- [B3実装差分・測定設計](b3_classical_preparation.md)
- [C1 provider・cache・価格仕様](c1_native_preparation.md)
- [費用条件と証拠の対応](cost_evidence_matrix.md)
- [未採択の実行契約候補](execution_contract_candidate.md) / [machine候補](execution_contract_candidate.json)
- [Decision Handoff](handoff.md)
- [補助SOURCE](../../../src/trottertracks/algorithm_codesign/g6_w2c_cost_preparation.py) / [人工テスト](../../../tests/tracks/algorithm_codesign/test_g6_w2c_cost_preparation.py)
- [結果schema](schemas/cost_record.schema.json) / [provider価格schema](schemas/provider_price.schema.json) / [候補schema](schemas/execution_candidate.schema.json)
- [人工テスト結果](artificial_test_results.json)、[技術検査・保全](verification.json)、[固定証拠索引](evidence_index.json)、[公開照合](publication_receipt.json)

SOURCEは標準libraryのみ。旧G7/F3/nativeモジュールをimportせず、人工Fraction・mock generator属性・fake clock・private temp textだけで**人工テスト48件PASS**。既存科学SOURCEやテストは変更せず、旧科学実験を再実行していない。実G7での数値correctness、runtime、cache coverage、platform/physical errorは未検証。

専用branch `track-b-g6-w2c-b3-c1-preparation-20261019`、基点`864097c1285f1f06399040cf0f7c0f42334d9252`の独立checkoutで作業。最新版共通指示2026-10-17の同一bytes・§8.4を確認した。CPU32、available RAM約49 GiB、/tmp空き約377 GiBの観測時点で軽量な人工検査のみ実施。共有設定/venv/cache/他jobは変更しない。旧固定resource契約も変更しない。

**mandatory STOP。** 次はユーザーが未決の費用estimand・実行条件・物理providerを判断する。契約候補の正式採択と、B3/C1科学実行の明示承認は別々に必要。候補は`adopted=false`、`contract_grants_execution=false`。authorization/grant/marker/claimは発行していない。

<!-- FIXED_REFERENCES -->

[A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/864097c1285f1f06399040cf0f7c0f42334d9252/docs/research/track_b_g6_w2c_total_cost_conditions_20261018/README.md
