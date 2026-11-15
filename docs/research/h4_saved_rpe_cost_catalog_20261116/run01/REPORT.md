# H4 保存RPE評価結果の比較

結論の範囲：**指定した保存候補集合内の最安適合点**。
保存Q3判定と条件付き価格の比較であり、全候補最適性・正式E[C]/G・方式winnerの認定は行わない。

|方式|公式 / LD|要求幅|最安適合の条件付き費用 RZ|point ID|
|---|---|---|---:|---|
|FULL_PF|S2 / 10|16h/7|559,457,386.00|54f360874da5113ae6efd9c0|
|TRUNCATED_PF|S2 / 7|16h/7|395,962,678.00|f5b43c8076101f52353bb98f|
|PR_PAULI|S2 / 3|1h|541,936,295.95|06f67f6f4ae4a7390d403c96|
|PR_DF|S2 / 7|1h|1,235,340,731.91|ab1d8980b9d136b73e43b0f3|
|FULL_H|pure RTE / 0|—|1,246,382,808,920.55|62aee6d389d5ebd3045509b0|

|候補|方式|LD|公式|判定|費用 RZ|X/Y配分|
|---|---|---:|---|---|---:|---|
|FULL_PF_S2_16h_over_7__BASELINE_N0|FULL_PF|10|2nd|適合|838,693,580.00|[59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 15, 11] / [59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 15, 11]|
|FULL_PF_Y4_16h_over_3__BASELINE_N0|FULL_PF|10|4th|適合|1,089,674,842.00|[59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 15, 11] / [59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 15, 11]|
|FULL_PF_M5_16h__BASELINE_N0|FULL_PF|10|4th(m5_best)|適合|1,343,396,772.00|[59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 15, 11] / [59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 15, 11]|
|B0I_P7_S2_16h_over_7__BASELINE_N0|TRUNCATED_PF|7|2nd|適合|593,555,352.00|[59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 15, 11] / [59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 15, 11]|
|B0I_P7_Y4_16h_over_3__BASELINE_N0|TRUNCATED_PF|7|4th|適合|765,902,758.00|[59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 15, 11] / [59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 15, 11]|
|B0I_P7_M5_16h__BASELINE_N0|TRUNCATED_PF|7|4th(m5_best)|適合|951,282,896.00|[59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 15, 11] / [59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 15, 11]|
|FULL_PF_S2_16h_over_7__A_SHAPE|FULL_PF|10|2nd|適合|622,182,168.00|[46, 43, 40, 37, 33, 30, 27, 24, 21, 18, 14, 11, 8] / [46, 43, 40, 37, 33, 30, 27, 24, 21, 18, 14, 11, 8]|
|FULL_PF_S2_16h_over_7__B_PRICE|FULL_PF|10|2nd|適合|615,304,502.00|[59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 13, 4] / [59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 13, 4]|
|FULL_PF_Y4_16h_over_3__A_SHAPE|FULL_PF|10|4th|適合|628,216,816.00|[36, 33, 31, 28, 26, 23, 21, 18, 16, 14, 11, 9, 6] / [36, 33, 31, 28, 26, 23, 21, 18, 16, 14, 11, 9, 6]|
|FULL_PF_Y4_16h_over_3__B_PRICE|FULL_PF|10|4th|適合|619,717,524.00|[59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 7, 2] / [59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 7, 2]|
|FULL_PF_M5_16h__A_SHAPE|FULL_PF|10|4th(m5_best)|適合|628,785,102.00|[30, 28, 26, 24, 22, 20, 17, 15, 13, 11, 9, 7, 5] / [30, 28, 26, 24, 22, 20, 17, 15, 13, 11, 9, 7, 5]|
|FULL_PF_M5_16h__B_PRICE|FULL_PF|10|4th(m5_best)|モデル不適合|619,784,264.00|[59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 13, 5, 1] / [59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 13, 5, 1]|
|B0I_P7_S2_16h_over_7__A_SHAPE|TRUNCATED_PF|7|2nd|適合|440,329,944.00|[46, 43, 40, 37, 33, 30, 27, 24, 21, 18, 14, 11, 8] / [46, 43, 40, 37, 33, 30, 27, 24, 21, 18, 14, 11, 8]|
|B0I_P7_S2_16h_over_7__B_PRICE|TRUNCATED_PF|7|2nd|適合|435,481,090.00|[59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 13, 4] / [59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 13, 4]|
|B0I_P7_Y4_16h_over_3__A_SHAPE|TRUNCATED_PF|7|4th|適合|444,807,010.00|[37, 34, 32, 29, 27, 24, 22, 19, 17, 14, 11, 9, 6] / [37, 34, 32, 29, 27, 24, 22, 19, 17, 14, 11, 9, 6]|
|B0I_P7_Y4_16h_over_3__B_PRICE|TRUNCATED_PF|7|4th|適合|435,641,568.00|[59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 7, 2] / [59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 7, 2]|
|B0I_P7_M5_16h__A_SHAPE|TRUNCATED_PF|7|4th(m5_best)|適合|445,015,812.00|[30, 28, 26, 24, 22, 19, 17, 15, 13, 11, 9, 7, 5] / [30, 28, 26, 24, 22, 19, 17, 15, 13, 11, 9, 7, 5]|
|B0I_P7_M5_16h__B_PRICE|TRUNCATED_PF|7|4th(m5_best)|モデル不適合|438,954,868.00|[59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 13, 5, 1] / [59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 13, 5, 1]|
|P7_S2_LOCAL_R12_D1 (m12=3)|TRUNCATED_PF|7|2nd|適合|415,721,884.00|[59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 13, 3] / [59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 13, 3]|
|P7_S2_LOCAL_R12_D2 (m12=2)|TRUNCATED_PF|7|2nd|適合|395,962,678.00|[59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 13, 2] / [59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 13, 2]|
|P7_S2_M12_ONE_FIXED (m12=1)|TRUNCATED_PF|7|2nd|モデル不適合|376,203,472.00|[59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 13, 1] / [59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 13, 1]|
|FPN_B_LD2|PR_PAULI|2|2nd|モデル不適合|600,045,823.85|[48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 7, 1, 1] / [48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 7, 1, 1]|
|OWN_B_LD2|PR_PAULI|2|2nd|適合|2,442,845,584.51|[48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 16, 13, 5] / [48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 16, 13, 5]|
|PR_DF_LD1|PR_DF|1|2nd|適合|261,314,830,475.56|[64, 60, 56, 52, 47, 43, 38, 34, 30, 25, 21, 17, 12] / [64, 60, 56, 52, 47, 43, 38, 34, 30, 25, 21, 17, 12]|
|LD1_A_SHAPE|PR_DF|1|2nd|適合|193,529,481,006.19|[48, 45, 42, 39, 35, 32, 29, 25, 22, 19, 16, 12, 9] / [48, 45, 42, 39, 35, 32, 29, 25, 22, 19, 16, 12, 9]|
|LD1_B_PRICE|PR_DF|1|2nd|適合|189,232,849,610.91|[64, 60, 56, 52, 47, 43, 38, 34, 30, 25, 21, 17, 7] / [64, 60, 56, 52, 47, 43, 38, 34, 30, 25, 21, 17, 7]|
|PR_DF_LD2|PR_DF|2|2nd|適合|63,914,076,114.98|[64, 60, 56, 52, 47, 43, 38, 34, 30, 25, 21, 17, 12] / [64, 60, 56, 52, 47, 43, 38, 34, 30, 25, 21, 17, 12]|
|LD2_A_SHAPE|PR_DF|2|2nd|適合|47,334,382,243.30|[48, 45, 42, 39, 35, 32, 29, 25, 22, 19, 16, 12, 9] / [48, 45, 42, 39, 35, 32, 29, 25, 22, 19, 16, 12, 9]|
|LD2_B_PRICE|PR_DF|2|2nd|適合|46,341,978,336.21|[64, 60, 56, 52, 47, 43, 38, 34, 30, 25, 21, 17, 7] / [64, 60, 56, 52, 47, 43, 38, 34, 30, 25, 21, 17, 7]|
|FULL_H_KAPPA2|FULL_H|0|None|適合|1,246,382,808,920.55|[96, 90, 84, 78, 71, 65, 57, 51, 45, 38, 32, 26, 18] / [96, 90, 84, 78, 71, 65, 57, 51, 45, 38, 32, 26, 18]|
|FULL_PF_S2_M12_TWO_FIXED (m12=2)|FULL_PF|10|2nd|適合|559,457,386.00|[59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 13, 2] / [59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 13, 2]|
|PAULI_LD3_VP|PR_PAULI|3|2nd|適合|1,051,862,454.81|[48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 16, 13, 9] / [48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 16, 13, 9]|
|FULL_PF_S2_M12_ONE_FIXED (m12=1)|FULL_PF|10|2nd|モデル不適合|531,533,828.00|[59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 13, 1] / [59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 13, 1]|
|B0I_P7_Y4_32h_over_5|TRUNCATED_PF|7|4th|モデル不適合|—|[59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 7, 2] / [59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 7, 2]|
|LD7_V0|PR_DF|7|2nd|適合|1,668,829,456.07|[64, 60, 56, 52, 47, 43, 38, 34, 30, 25, 21, 17, 12] / [64, 60, 56, 52, 47, 43, 38, 34, 30, 25, 21, 17, 12]|
|LD7_A_SHAPE|PR_DF|7|2nd|適合|1,235,340,731.91|[48, 45, 42, 39, 35, 32, 29, 25, 22, 19, 16, 12, 9] / [48, 45, 42, 39, 35, 32, 29, 25, 22, 19, 16, 12, 9]|
|LD7_B_PRICE|PR_DF|7|2nd|適合|1,240,709,696.98|[64, 60, 56, 52, 47, 43, 38, 34, 30, 25, 21, 14, 5] / [64, 60, 56, 52, 47, 43, 38, 34, 30, 25, 21, 14, 5]|
|B0I_P7_Y4_32h_over_5__A_SHAPE|TRUNCATED_PF|7|4th|Q3適合・価格待ち|—|[37, 34, 32, 29, 27, 24, 22, 19, 17, 14, 11, 9, 6] / [37, 34, 32, 29, 27, 24, 22, 19, 17, 14, 11, 9, 6]|
|P7__A_SHAPE|PR_PAULI|3|2nd|モデル不適合|389,727,554.49|[21, 19, 18, 17, 15, 14, 12, 11, 10, 8, 7, 5, 3] / [21, 19, 18, 17, 15, 14, 12, 11, 10, 8, 7, 5, 3]|
|P7__B_PRICE|PR_PAULI|3|2nd|モデル不適合|395,206,798.24|[48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 15, 4, 1] / [48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 15, 4, 1]|
|FULL_PF__A_SHAPE|PR_PAULI|3|2nd|モデル不適合|510,918,151.05|[26, 24, 23, 21, 19, 18, 16, 14, 12, 10, 8, 7, 4] / [26, 24, 23, 21, 19, 18, 16, 14, 12, 10, 8, 7, 4]|
|FULL_PF__B_PRICE|PR_PAULI|3|2nd|適合|541,936,295.95|[48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 16, 8, 2] / [48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 16, 8, 2]|
|LD9_V0|PR_DF|9|2nd|Q3適合・価格待ち|—|[64, 60, 56, 52, 47, 43, 38, 34, 30, 25, 21, 17, 12] / [64, 60, 56, 52, 47, 43, 38, 34, 30, 25, 21, 17, 12]|

重複除去：4件。詳細・固定SOURCE・26軸の価格対応はCATALOG.jsonを参照。
Q3適合・価格待ちは費用nullで選択対象外。信号だけの旧状態はsignal_only_historyへ保持する。
Full-Hは同一固定配分のQ3を照合した経験的m0＋24モデル価格のシナリオ。
PRの事前固定予算・旧PF基準はhistorical_*に元SOURCEとともに保持し、新PF最安値で再計算しない。
新規signal・sampling・compile・fit・配分探索・Q3はすべて0。

## 比較条件

```json
{
  "binding": {
    "system": "H4",
    "snapshot": "12219b7d14dc2a275853314269e5e8628005ed72b42d16bc4d3831c8ea515c96",
    "hamiltonian": "c1a6c4281f11bba90df2fd18a998bdec96bed00379116537d22faf548c525ea0",
    "state": "9bcc0b0ecf44440c4411bcb76d652d78ff0595f036144580660c56c47c78e80b",
    "basis": "STO-3G",
    "geometry": "linear_1Angstrom",
    "rank_policy": "tol_only_1e-8"
  },
  "accuracy": {
    "estimator": "consecutive_dyadic_nearest_branch_principal_energy_total_v1",
    "target": "SAVED_DF10_SOLVER_ENERGY_CONDITIONAL_PROXY",
    "loss": "ALL_EVENT_SQUARED_ENERGY_ERROR",
    "epsilon_E_decimal": "0.00159360010199040",
    "MSE_upper": {
      "numerator": 15136964828156312001,
      "denominator": 5960464477539062500000000
    },
    "scope": "MODEL_MSE_FILTER_ONLY"
  },
  "h": {
    "numerator": 3602879701896397,
    "denominator": 36028797018963968
  },
  "M": 12,
  "P": 0,
  "price_quantity": "MEASURED_FULL_CONTROLLED_HADAMARD_WRAPPER_COMPILED_RZ_PER_SHOT",
  "interpretation": "CONDITIONAL_RAW_IID_MODEL_MSE_AND_SAVED_PRICE_SCENARIOS_NOT_FORMAL_G",
  "compiler_profile": {
    "compiler": "1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b",
    "control": "symmetric_directional",
    "wrapper": "full_Hadamard_with_measurement",
    "P": 0,
    "qiskit_version": "1.3.0",
    "basis_gates": [
      "rz",
      "sx",
      "x",
      "cx"
    ],
    "optimization_level": 1,
    "seed_transpiler": 17
  },
  "energy_prior_ha": [
    -4.994249624417489,
    -2.0985459369977186
  ],
  "target": {
    "kind": "SAVED_FULL_DF_H_SOLVER_CONDITIONAL_PROXY",
    "energy_ha": -2.1663874486347634
  }
}
```

## 固定入力一覧

- `b4296b110b129d226dbef2743c14f725a27c3bf6:docs/research/pf6_cost_budget_q3_results_20261115/SUMMARY.json`
- `cbc471c913072de45c25a0ed22d51dff8a9b7034:docs/research/p7_s2_m12_one_results_20261116/SUMMARY.json`
- `738df226232e539cc084dfad2aed2feae82f3073:docs/research/pauli_ld2_reallocation_q3_results_20261115/SUMMARY.json`
- `cdbce383f9944b7f9da1cbbd7851406dbbaae88a:docs/research/df_head_pauli_tail_ld3_signals_20261116/MEAN_HANDOFF.json`
- `8ec7c882806809460d38a638a891bafcf18fafd1:docs/research/d_pr_df_ld_split_q3_resume_20261116/SUMMARY.json`
- `6ad23def093584d834f1de7731312d958727f67c:docs/research/full_h_m0_zero_zz_admission_total_20261116/run01/RESULT.json`
- `a006b8e15f86ee42ed66f89fbd1def721d027e87:docs/research/full_pf_s2_m12_two_results_20261116/C_RESULT_JOIN.json`
- `a006b8e15f86ee42ed66f89fbd1def721d027e87:docs/research/full_pf_s2_m12_two_results_20261116/SUMMARY.json`
- `2de33a4f4daeef7ca899dc98689974bb0eaadaf4:docs/research/pauli_ld3_vp_q3_results_20261116/C_RESULT_JOIN.json`
- `0defa286c53a1b9acf513be453907bcad83102b3:docs/research/d_pr_df_ld7_v0_results_20261116/SUMMARY.json`
- `7349f6f9b1dad3e67a77ffe948bfbf347b5c23a7:docs/research/full_pf_s2_m12_one_results_20261116/C_RESULT_JOIN.json`
- `7349f6f9b1dad3e67a77ffe948bfbf347b5c23a7:docs/research/full_pf_s2_m12_one_results_20261116/SUMMARY.json`
- `8a2f5c46d201b26bdaf2aad7ca556e31083779aa:docs/research/d_p7_y4_coarse_q3_recovery_20261116/C_RESULT_JOIN.json`
- `f25a05f8a92aab43cfbe30fc80b974054156aac1:docs/research/a_pauli_ld3_cal_price_20261116/HANDOFF_BC.json`
- `fd010b111e95411702be51adc73e56b0bba6c093:docs/research/d_pr_df_ld7_price_results_20261116/C_HANDOFF.json`
- `fd010b111e95411702be51adc73e56b0bba6c093:docs/research/d_pr_df_ld7_price_results_20261116/SUMMARY.json`
- `8855f420a9fe2577efc77f099c5a39ab0cb9b7e6:docs/research/d_p7_y4_coarse_a_shape_q3_20261116/C_RESULT_JOIN.json`
- `fb4d78781e2a4c63c9d334af246807785e669d6c:docs/research/pauli_ld3_pf_budget_q3_results_20261116/C_COMPARISON_JOIN.json`
- `778118926c85158f947928c1e744d565464aab17:docs/research/d_pr_df_ld9_v0_results_20261116/C_HANDOFF.json`

PR-Pauli LD3 VPは保存Q3とAの26軸モデル価格を接続した条件付き参照総額。直接取得は別較正cellの2pathのみ、qualified_for_candidate=falseを保持する。

LD7_A_SHAPEの費用の95.9453590253993779917152%はq>64のm7〜m12へのモデル外挿（元SUMMARYの保存内訳）。qualified_for_candidate=false。価格モデル・配分・Q3・較正receiptの固定locatorはCATALOG.jsonに保持。

PR-Pauli FULL_PF__B_PRICEの保存費用差（T−PF基準）：Full PF S2 -17,521,090.05 RZ、p7 S2 145,973,617.95 RZ。全26価格はモデル参照、qualified_for_candidate=false。

再実行コマンドと追加方法は親directoryのHANDOFF.mdを参照。
