# A → C：Full-H κ2 P-1の単一path direct価格（新ZZ、m3、X/Y）

経緯と根拠は[REPORT](REPORT.md)にある。原run02は`STOP_WIRING_CHECK`のままで、修正SOURCEによる事後再検証がPASSしている。

## 引き渡すもの

| 項目 | 内容 |
|---|---|
| 価格record | [PRICES.json](PRICES.json)。X/Y各1件、`FULL_WRAPPER_MEAN_COMPILED_PRICE`の`mean.rz_count`＝92,448（`metrics`にCX・depth・sizeも含む） |
| 分類 | `DIRECT_COMPILED_FINITE_SAMPLE`／`FINITE_COHORT_ONLY`、`n_paths=1`、`law_expectation=false`、`record_schema=D_PRICE_RECORD_B_KINDS_V2` |
| sidecar | [SIDECARS.json](SIDECARS.json)。`implementation`（FULL_H_RTE、native・wrapper・scalarのSOURCE、新ZZの`rte_lowering_source`、`ONCE_AT_TOTAL_T`、measurement）と`implementation_evidence` |
| implementation evidence | [IMPLEMENTATION_EVIDENCE.json](IMPLEMENTATION_EVIDENCE.json)。固定commit `6d188defdfccd52194b826610bf77a333869ec11`、selector `3X`・`3Y` |
| guard結果 | [GUARD_CHECK.json](GUARD_CHECK.json)。Cの`c628e9e8` guardを無変更で実行し、`IMPLEMENTATION_POLICY_CHECKED_NOT_SCIENTIFIC_QUALIFICATION`、2件 |
| configuration | `54ac3eb38de5f90a77e9921100576c89f6c870f68384f4c7d9efe61b2c80c07e`（κ2、K2、全13 round） |
| round | m3のみ。physical candidateは`FULL_H_4de6130f0d3a66e5d695` |
| cohort | `D_FULL_H_KAPPA2_NEW_ZZ_M3_WIRING_V1` |
| identity・law_binding | canonical row（`I.price_requests`）のものをそのまま使用 |

## Cでの受入確認

- **cohort**：`cohorts={(m3のcandidate id, 'X'|'Y'): 'D_FULL_H_KAPPA2_NEW_ZZ_M3_WIRING_V1'}`を指定する。
- **policy**：受入確認に使ったpolicyは`D_FULL_H_KAPPA2_NEW_ZZ_P1_IMPLEMENTATION_ACCEPTANCE_V1`（GUARD_CHECKの`result.policy`）である。比較scenarioとしてのpolicyはユーザーが選ぶ。
- **lowering**：同じpolicyにPR（DF）を含める場合も、loweringは新ZZに揃える必要がある（旧ZZとの混在はguardが拒否する）。

## 使い方の制限

- **単一trajectory・単一seedのdirect価格**であり、finite cohortである。
- 次のものとして扱わない：
  - Full-Hの\(\mathbb E[C]\)
  - 平均event単価
  - `price_model_validation`の較正点・holdout
  - P-2の入力
  - G
  - 方式比較
- Cの価格計画（PRICE_VALIDATION_PLAN）でも、P-1は配線確認の位置付けである。
- `law_context.price_law`は`PCG64_numpy_float64_CDF_saved_cohort`、`ideal_sampler_distance_qualified=false`。
- axisごとのcompile時間は記録がない（UNKNOWN）。

## 未実施（承認範囲外）

P-2、S-2、run03、N・MSE・DP・G、価格modelのfit、比較policyの採択。
