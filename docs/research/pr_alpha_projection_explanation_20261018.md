# PR資源表のnormalizationと親会計の説明訂正

2026-10-18 JST。α監査 `e62ddb59173d100c113be7a33581515d78f152ee` のD01/A05への説明追補。保存されたshot・費用・順位、科学SOURCEと契約は変更しない。

## D01 H4 geometry

H4_GEOMETRY/H4_GEOMETRY_B0Iは、保存`normalization`を直接shot式へ使う旧geometry契約である。各軸α=.025、h=ε/√2−保存axis bias、N=ceil(2 B² log80/h²)。deterministicではB=1。native H4/H6/H8にある経験的uの明示差引きやsaved log_B_marginをこの系列へ追加しない。

[原signal SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d0726e946f89f8c30a08d383fb8ff48b363629ec/src/trottertracks/resource_applicability/h4_geometry/signal.py#L200)と[B0I保存値会計](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/98cdb35cd08469714f20d6cc3b982d9c80908f46/artifacts/resource_applicability/track_a_h4_b0i_map_results/2026-10-14/analyze_saved_results.py#L44)が根拠である。

横断inventoryの2行の`normalization_policy`だけをJSON/CSVで訂正した。[修正前後・hash・根拠](../../artifacts/resource_applicability/pr_alpha_followup_preparation/2026-10-18/stage_d_changes.json)、[diff](../../artifacts/resource_applicability/pr_alpha_followup_preparation/2026-10-18/stage_d_diff.patch)。旧artifact receiptは当時の固定commitの記録として残し、今回の説明訂正は別receiptで追跡する。未保存event/component履歴と既知172候補のnormalization来歴不足（A04）は未解決のまま。

## A05 H8通常biasの親会計

H8主要6軸反映結果 `96e69a01d1fae7a402aac2c4efa570a59f8cc7a0` の通常biasは簡略projectionで、α・証拠分類・shot状態を単独では保持しない。参照を次のようにつなぐ。

|役割|固定参照と内容|
|---|---|
|6軸反映の実行SOURCE|[aa984d98…のcontractと入力join](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aa984d98a7b438dc953a829a24fe3d7edc9d0a6b/scripts/h8_six_axis_reaccount_contract.py#L273)。cell_id/ε/factorをキーに保存accountingからshotを参照する|
|通常projectionの親|[通常Phase Cの保存行](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6dc5bf0de01a0aa23cafe0ff022c0cd21fbf1086/artifacts/resource_applicability/track_a_h8_phase_c_run01/2026-10-17/accounting_rows_v1.jsonl.gz)。以下の元accountingへの参照も必要|
|元B0/full PF/確認PR|[H8 D0 accounting](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3d910ee1de4597b8895e8506500553dbad5a5085/artifacts/resource_applicability/track_a_h8_d0_saved_value_execution_v1/2026-10-15/candidate_accounting_v1.json)。H8_EXPLORATION92の登録detとH8_CONFIRMATION_UNION6の確認n32|
|旧B0I|[case accounting](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f79257f432bc7e0317f122090f62fec379e2a1b/artifacts/resource_applicability/track_a_h8_native_b0i_full_family_result_v1/2026-10-16/case_accounting_v1.json)|
|追加152候補|[Phase A保存signalのprecision_rows](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/01309059636f2432e29b663bcd2eb0e400f5ae0a/artifacts/resource_applicability/track_a_h8_phase_a_run01/2026-10-16/H8_signal_complete_v1.json.gz)|
|非相殺の別親|[23d46e48…の非相殺accounting](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23d46e489af054735d9d328aad0342ecdbddf305/artifacts/resource_applicability/track_a_h8_noncancellation_cost_run01/2026-10-17/accounting_rows_v1.jsonl.gz)。通常signed biasと混合しない|
|入力のbytes/SHA|[aa984d98…の入力manifest](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aa984d98a7b438dc953a829a24fe3d7edc9d0a6b/references/h8_six_axis_reaccount_inputs_v1.json)|
|共通shot式|[監査済み凍結D0関数](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fa75c7997e135da7b7a0c4e23949e47252bd7a2e/scripts/d0_frozen_shot_rule_v1.py)|

各軸α=.025、許容幅ε/√2で、biasと宣言allowanceを引いたheadroomとB_budgetにshot式を適用する。通常biasはsigned成分を合成した軸誤差の絶対値、経験的非相殺は成分絶対値和を使う別診断で、後者も厳密bias certificateではない。

`EMPIRICAL`は経験的allowanceに条件付く会計。`CERTIFIED`は呼出側の証拠分類であり、shot関数自身の`certificate_verified_by_this_function`はfalse。UNKNOWN、UNDETERMINED、LOG_DOMAIN_ONLY等は親accountingから判断し、簡略表の欠落欄を認証済みや有限費用に読み替えない。

1 shotの補正倍率B_corrとshot予算の上界B_budgetを区別する。αは量子測定統計の失敗確率、n32のpaired SEはランダム回路費用平均の有限標本不確かさである。どちらも他方の保証にならない。

6軸の実測RZを優先する行と、固定model RZを使う行は保存`axis_costs`/`cost_status`の区分を維持する。全184候補を実測したとの主張や、PR母平均winnerへの昇格は行わない。本追補による数値解析・科学条件変更は0。

Stage Dは説明訂正のみ完了。B-1/B-2の数値実行は別承認である。
