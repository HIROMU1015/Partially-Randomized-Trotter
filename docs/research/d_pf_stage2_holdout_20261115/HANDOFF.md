# C・統合担当への引渡し：決定論PF m6独立holdout

campaign PASS。予定12、保存再利用0、新規compile12、モデル個別PASS/FAIL/UNKNOWN=12/0/0。取得/作用gate/実装guardは全12 wrapper完了。追加科学計算には進まずmandatory STOP。

正本は [RESULT.json](run01/RESULT.json) と12行 [HOLDOUT.csv](run01/HOLDOUT.csv)。予測は元C JSONのexact値、観測分母で1/50判定。各rowにmodel_group、m6 physical ID、axis、q、凍結予測、観測RZ、exact相対誤差、判定、receipt commit/path/selectorを保存した。

凍結SOURCE：`12372fa6b53c878dc5dd124e036572ba0cd82061:docs/research/pf_stage1_model_freeze_20261115/run01/FROZEN_MODELS.json`。SHA-256 `845c705082565a2b44d198bddb3034497620d991a00604454394ae5acda9651a`。D readerの開封前確認は [FREEZE_PREOPEN_V2](FREEZE_PREOPEN_V2.json)。Stage 1入力はb190bd815の索引から元C_INPUTSを読み、不変を確認した。Dは再fit・予測再計算を行っていない。policyは91365dcf、RELATIVE/observed/1/50、全12個別PASSがcampaign成功。自動refit・閾値緩和・軸補完なし。

## 取得SOURCE・公開receipt

Full PF private：`/home/AbeHiromu/projects/d-full-pf-shortlist-20261114-private/holdout/run01`。p7 private：`/home/AbeHiromu/projects/d-truncated-p7-t3-20261114-private/holdout/run01`。各job・cost・compiled_cache/QPYを分離して保持する。

| route | 実行SOURCE | raw価格公開 | bound公開 |
|---|---|---|---|
| FULL_PF | `e276ab70fdd6ff988928703ec79fd3a9aa8c1603` | `cd25fd4cb7c655021a50e1be52249fdcf97f7f42` | `f9755e73dacb6f9efbffd452f2e1ac550eed22ec` |
| TRUNCATED_PF | `f9755e73dacb6f9efbffd452f2e1ac550eed22ec` | `c2f6f5078f5d1bb5dea1a2f286ef8eb9b0570d40` | `b2aef9faff475e65da40fef9764d21ce35386c7c` |

**FULL_PF**

- [RESULT.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd25fd4cb7c655021a50e1be52249fdcf97f7f42/docs/research/d_full_pf_shortlist_prices_20261114/holdout/run01/RESULT.json)
- [PRICES.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd25fd4cb7c655021a50e1be52249fdcf97f7f42/docs/research/d_full_pf_shortlist_prices_20261114/holdout/run01/PRICES.json)
- [C_INPUTS.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd25fd4cb7c655021a50e1be52249fdcf97f7f42/docs/research/d_full_pf_shortlist_prices_20261114/holdout/run01/C_INPUTS.json)
- [IMPLEMENTATION_EVIDENCE.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd25fd4cb7c655021a50e1be52249fdcf97f7f42/docs/research/d_full_pf_shortlist_prices_20261114/holdout/run01/IMPLEMENTATION_EVIDENCE.json)
- [SIDECARS_UNBOUND.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd25fd4cb7c655021a50e1be52249fdcf97f7f42/docs/research/d_full_pf_shortlist_prices_20261114/holdout/run01/SIDECARS_UNBOUND.json)
- [SIDECARS.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f9755e73dacb6f9efbffd452f2e1ac550eed22ec/docs/research/d_full_pf_shortlist_prices_20261114/holdout/run01/SIDECARS.json)
- [GUARD_CHECK.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f9755e73dacb6f9efbffd452f2e1ac550eed22ec/docs/research/d_full_pf_shortlist_prices_20261114/holdout/run01/GUARD_CHECK.json)

**TRUNCATED_PF**

- [RESULT.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c2f6f5078f5d1bb5dea1a2f286ef8eb9b0570d40/docs/research/d_truncated_p7_t3_prices_20261114/holdout/run01/RESULT.json)
- [PRICES.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c2f6f5078f5d1bb5dea1a2f286ef8eb9b0570d40/docs/research/d_truncated_p7_t3_prices_20261114/holdout/run01/PRICES.json)
- [C_INPUTS.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c2f6f5078f5d1bb5dea1a2f286ef8eb9b0570d40/docs/research/d_truncated_p7_t3_prices_20261114/holdout/run01/C_INPUTS.json)
- [IMPLEMENTATION_EVIDENCE.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c2f6f5078f5d1bb5dea1a2f286ef8eb9b0570d40/docs/research/d_truncated_p7_t3_prices_20261114/holdout/run01/IMPLEMENTATION_EVIDENCE.json)
- [SIDECARS_UNBOUND.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c2f6f5078f5d1bb5dea1a2f286ef8eb9b0570d40/docs/research/d_truncated_p7_t3_prices_20261114/holdout/run01/SIDECARS_UNBOUND.json)
- [SIDECARS.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2aef9faff475e65da40fef9764d21ce35386c7c/docs/research/d_truncated_p7_t3_prices_20261114/holdout/run01/SIDECARS.json)
- [GUARD_CHECK.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2aef9faff475e65da40fef9764d21ce35386c7c/docs/research/d_truncated_p7_t3_prices_20261114/holdout/run01/GUARD_CHECK.json)

元cohort・H/state/snapshot・scalarの量と一回適用・native/measurement/compilerを保持し、元Stage 1・C凍結JSONを改変していない。各routeのSOURCEとpublic/private holdout/run01を分離した。SOURCE/実装不明のUNKNOWNを2%超過FAILへ置換しない。

ロードマップへの反映対象はm6独立holdoutの結果のみ。m7〜m12の外挿、追加compile、MSE/N/G、winnerや正式な全RPE最適性を認定しない。元モデルの資格flag・systematic/covarianceを自動昇格しない。REPORTに資源・preflight・残る不足を記載した。mandatory STOP。
