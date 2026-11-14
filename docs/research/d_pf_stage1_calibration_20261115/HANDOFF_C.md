# C向け：決定論PF Stage 1較正価格

新規20＋元保存4＝24較正価格、12 group、各m4/m5の2点。fit・係数凍結・HOLDOUT・N/GはDでは実行していない。mandatory STOP。

正本入口は[C_HANDOFF_INDEX.json](C_HANDOFF_INDEX.json)。各groupの`C_inputs_source`から既存C_INPUTSのgroupを読み、`saved_records[*].record`＋`new_records`を使用する。schemaは元C PRICE_CALIBRATION_RECORD_V1、featuresはone/q、roleは全てCALIBRATION、kindはDIRECT_COMPILED_DETERMINISTIC。12 groupを個別に処理し、X/Yやroute・公式・幅を平均/統合しない。

## 正確なmodel_group

- `FULL_PF|2nd|16/7|X`：m4 q=7 / RZ=54844、m5 q=14 / RZ=109381。
- `FULL_PF|2nd|16/7|Y`：m4 q=7 / RZ=54844、m5 q=14 / RZ=109381。
- `FULL_PF|4th(m5_best)|16|X`：m4 q=1 / RZ=85960、m5 q=2 / RZ=171613。
- `FULL_PF|4th(m5_best)|16|Y`：m4 q=1 / RZ=85960、m5 q=2 / RZ=171613。
- `FULL_PF|4th|16/3|X`：m4 q=3 / RZ=70912、m5 q=6 / RZ=141517。
- `FULL_PF|4th|16/3|Y`：m4 q=3 / RZ=70912、m5 q=6 / RZ=141517。
- `TRUNCATED_PF|2nd|16/7|X`：m4 q=7 / RZ=38898、m5 q=14 / RZ=77489。
- `TRUNCATED_PF|2nd|16/7|Y`：m4 q=7 / RZ=38898、m5 q=14 / RZ=77489。
- `TRUNCATED_PF|4th(m5_best)|16|X`：m4 q=1 / RZ=60950、m5 q=2 / RZ=121593。
- `TRUNCATED_PF|4th(m5_best)|16|Y`：m4 q=1 / RZ=60950、m5 q=2 / RZ=121593。
- `TRUNCATED_PF|4th|16/3|X`：m4 q=3 / RZ=49924、m5 q=6 / RZ=99541。
- `TRUNCATED_PF|4th|16/3|Y`：m4 q=3 / RZ=49924、m5 q=6 / RZ=99541。

## 固定公開先

**FULL_PF**：execution SOURCE `91365dcf0610362a96aab101fbe0ec0e78f58af1`。

- [C_INPUTS.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1078d23621566728521d8056fead17118244be42/docs/research/d_full_pf_shortlist_prices_20261114/calibration/run01/C_INPUTS.json)
- [PRICES.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1078d23621566728521d8056fead17118244be42/docs/research/d_full_pf_shortlist_prices_20261114/calibration/run01/PRICES.json)
- [RESULT.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1078d23621566728521d8056fead17118244be42/docs/research/d_full_pf_shortlist_prices_20261114/calibration/run01/RESULT.json)
- [IMPLEMENTATION_EVIDENCE.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1078d23621566728521d8056fead17118244be42/docs/research/d_full_pf_shortlist_prices_20261114/calibration/run01/IMPLEMENTATION_EVIDENCE.json)
- [SIDECARS_UNBOUND.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1078d23621566728521d8056fead17118244be42/docs/research/d_full_pf_shortlist_prices_20261114/calibration/run01/SIDECARS_UNBOUND.json)
- [SIDECARS.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a393a65cc2f00cdad596e11b9910bae0394e2407/docs/research/d_full_pf_shortlist_prices_20261114/calibration/run01/SIDECARS.json)
- [GUARD_CHECK.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a393a65cc2f00cdad596e11b9910bae0394e2407/docs/research/d_full_pf_shortlist_prices_20261114/calibration/run01/GUARD_CHECK.json)

**TRUNCATED_PF**：execution SOURCE `a393a65cc2f00cdad596e11b9910bae0394e2407`。

- [C_INPUTS.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a04d40fe4d8f4bcf8083e4feebdf1535b5e7f5cd/docs/research/d_truncated_p7_t3_prices_20261114/calibration/run01/C_INPUTS.json)
- [PRICES.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a04d40fe4d8f4bcf8083e4feebdf1535b5e7f5cd/docs/research/d_truncated_p7_t3_prices_20261114/calibration/run01/PRICES.json)
- [RESULT.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a04d40fe4d8f4bcf8083e4feebdf1535b5e7f5cd/docs/research/d_truncated_p7_t3_prices_20261114/calibration/run01/RESULT.json)
- [IMPLEMENTATION_EVIDENCE.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a04d40fe4d8f4bcf8083e4feebdf1535b5e7f5cd/docs/research/d_truncated_p7_t3_prices_20261114/calibration/run01/IMPLEMENTATION_EVIDENCE.json)
- [SIDECARS_UNBOUND.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a04d40fe4d8f4bcf8083e4feebdf1535b5e7f5cd/docs/research/d_truncated_p7_t3_prices_20261114/calibration/run01/SIDECARS_UNBOUND.json)
- [SIDECARS.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ff582779ee36741407cac2d27e0f7b243088a0d/docs/research/d_truncated_p7_t3_prices_20261114/calibration/run01/SIDECARS.json)
- [GUARD_CHECK.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ff582779ee36741407cac2d27e0f7b243088a0d/docs/research/d_truncated_p7_t3_prices_20261114/calibration/run01/GUARD_CHECK.json)

## 元保存価格とpolicy

M5/m4のFull X/Y各85960、p7 X/Y各60950だけが保存再利用。EXPLORATION cohort・元price SOURCE `155529e20a8fac39e00786c40b57830b71203c4f` を保持する。C正本は `855573064b2c17cfe20eb94a01ece1a554141b0b:docs/research/truncated_p7_admission_pr_df_evidence_20261114/SAVED_RECEIPT_SOURCES.json`。Full admissionは2fa88065、p7入力/価格は43c15bfa・sidecar/guardは2b945397。各C_INPUTS.saved_records.referencesから元recordへ辿る。新cohortへ付け替えない。

新規FullはD_FULL_PF_SHORTLIST_DETERMINISTIC_V1、p7はD_TRUNCATED_P7_DETERMINISTIC_T3_V1。新10価格/routeのC guardは全件受理。旧SIDECARS_UNBOUNDはそのまま残し、新SIDECARSがraw公開commitへ結合する。原本を書き換えていない。

科学条件は `91365dcf0610362a96aab101fbe0ec0e78f58af1:scripts/pf_price_stages.py` のCOMPARISON_POLICY、値はindexにも収録。H-b＝RELATIVE、scale=observed、1/50、12モデル個別で全PASS、m6非較正、FAIL時に自動refit/閾値緩和しない。今回この判定を実行していない。過去C保存資料の未採択表記は当時の状態として保全し、現policyは今回の固定条件に従う。

次のC実fit/係数凍結は別途科学実行指示待ち。取得済みであること・guard受理と、モデル妥当性/長q外挿資格を分離する。HOLDOUT m6 12 compile、m7〜m12価格モデル、N/G・優劣は未評価。SOURCE/raw/保存cohort/STOPを維持する。
