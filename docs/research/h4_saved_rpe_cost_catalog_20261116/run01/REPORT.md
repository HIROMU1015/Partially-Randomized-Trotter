# H4 保存RPE評価結果の比較

結論の範囲：**指定した保存候補集合内の最安適合点**。
保存Q3判定と条件付き価格の比較であり、全候補最適性・正式E[C]/G・方式winnerの認定は行わない。

|方式|公式 / LD|要求幅|最安適合の条件付き費用 RZ|point ID|
|---|---|---|---:|---|
|FULL_PF|M5 / 10|16h|480,258,064.00|6531de11e60718dc79f7c12f|
|TRUNCATED_PF|Yoshida4 / 7|32h/5|372,035,410.00|bfc0c1081b78c5adcb12b43b|
|PR_PAULI|S2 / 5|1h|410,199,237.00|b36673acc8bb5d916b6b658b|
|PR_DF|S2 / 7|1h|1,235,340,731.91|ab1d8980b9d136b73e43b0f3|
|FULL_H|pure RTE / 0|—|1,246,382,808,920.55|62aee6d389d5ebd3045509b0|

現在の保存集合内最安同士の差（方式の費用−各基準、正なら基準より高額）：

|方式|−Full PF RZ|−B0I p7 RZ|
|---|---:|---:|
|FULL_PF|0.00|108,222,654.00|
|TRUNCATED_PF|-108,222,654.00|0.00|
|PR_PAULI|-70,058,827.00|38,163,827.00|
|PR_DF|755,082,667.91|863,305,321.91|
|FULL_H|1,245,902,550,856.55|1,246,010,773,510.55|

PR_PAULI_KAPPA_ABSOLUTE_2：501,233,190.43 RZ（絶対κ=2限定。元κの倍率×2や方式全体の最安とは別）。
point ID：fe8b73742b890af3820ca6f9。

LD5_KAPPA_X2_INTERRUPTED_BEST：探索方針変更による途中終了（INTERRUPTED_SEARCH_POLICY_CHANGE）。収束値ではない。
κ絶対値=26225228882306721296844821347007773673162352858535179072118456320/539496659426909122096437389293855970805348475396548904610992201、元κ倍率=2。
保存総額と26軸MODEL適用summaryを受入。個別26単価は未保存/null、全26軸外挿、qualified_for_candidate=false。
exact MSE上下界は公開hashと一致した最安点1件のOUTCOMEを保持し、再評価していない。

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
|B0I_P7_Y4_32h_over_5__A_SHAPE|TRUNCATED_PF|7|4th|適合|372,035,410.00|[37, 34, 32, 29, 27, 24, 22, 19, 17, 14, 11, 9, 6] / [37, 34, 32, 29, 27, 24, 22, 19, 17, 14, 11, 9, 6]|
|P7__A_SHAPE|PR_PAULI|3|2nd|モデル不適合|389,727,554.49|[21, 19, 18, 17, 15, 14, 12, 11, 10, 8, 7, 5, 3] / [21, 19, 18, 17, 15, 14, 12, 11, 10, 8, 7, 5, 3]|
|P7__B_PRICE|PR_PAULI|3|2nd|モデル不適合|395,206,798.24|[48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 15, 4, 1] / [48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 15, 4, 1]|
|FULL_PF__A_SHAPE|PR_PAULI|3|2nd|モデル不適合|510,918,151.05|[26, 24, 23, 21, 19, 18, 16, 14, 12, 10, 8, 7, 4] / [26, 24, 23, 21, 19, 18, 16, 14, 12, 10, 8, 7, 4]|
|FULL_PF__B_PRICE|PR_PAULI|3|2nd|適合|541,936,295.95|[48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 16, 8, 2] / [48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 16, 8, 2]|
|LD9_V0|PR_DF|9|2nd|Q3適合・価格待ち|—|[64, 60, 56, 52, 47, 43, 38, 34, 30, 25, 21, 17, 12] / [64, 60, 56, 52, 47, 43, 38, 34, 30, 25, 21, 17, 12]|
|LD7_P7_S2__A_SHAPE|PR_DF|7|2nd|モデル不適合|352,435,885.50|[15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 2] / [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 2]|
|LD7_P7_S2__B_PRICE|PR_DF|7|2nd|モデル不適合|390,742,769.68|[64, 60, 56, 52, 47, 43, 38, 34, 26, 10, 4, 1, 1] / [64, 60, 56, 52, 47, 43, 38, 34, 26, 10, 4, 1, 1]|
|LD7_FULL_PF_S2__A_SHAPE|PR_DF|7|2nd|モデル不適合|537,571,380.12|[22, 21, 19, 18, 16, 15, 13, 11, 10, 8, 7, 5, 4] / [22, 21, 19, 18, 16, 15, 13, 11, 10, 8, 7, 5, 4]|
|LD7_FULL_PF_S2__B_PRICE|PR_DF|7|2nd|モデル不適合|548,149,803.13|[64, 60, 56, 52, 47, 43, 38, 34, 30, 19, 7, 3, 1] / [64, 60, 56, 52, 47, 43, 38, 34, 30, 19, 7, 3, 1]|
|LD5__FULL_PF__B_PRICE|PR_PAULI|5|2nd|適合|541,743,905.57|[48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 16, 8, 2] / [48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 16, 8, 2]|
|LD5__VP|PR_PAULI|5|2nd|適合|872,241,518.14|[48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 16, 13, 9] / [48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 16, 13, 9]|
|FULL_PF_OPTIMIZER_4th(m5_best)_16h|FULL_PF|10|4th(m5_best)|適合|480,258,064.00|[20, 25, 26, 24, 16, 20, 17, 15, 12, 11, 8, 7, 2] / [20, 25, 26, 24, 16, 20, 17, 15, 12, 11, 8, 7, 2]|
|LD5_KAPPA_X2_INTERRUPTED_BEST|PR_PAULI|5|2nd|適合|410,470,823.86|[48, 45, 42, 39, 24, 27, 14, 20, 16, 16, 8, 7, 2] / [48, 45, 42, 39, 24, 27, 14, 20, 16, 16, 8, 7, 2]|
|LD5_S2_delta_h_kappa_x2__edd63e0befd3af591809|PR_PAULI|5|2nd|適合|410,199,237.00|[48, 45, 42, 39, 22, 27, 14, 20, 16, 16, 8, 7, 2] / [48, 45, 42, 39, 22, 27, 14, 20, 16, 16, 8, 7, 2]|
|LD5_S2_delta_h_kappa_abs_2__e3cb2cfda719117d6222|PR_PAULI|5|2nd|適合|501,233,190.43|[48, 45, 42, 39, 36, 32, 14, 21, 17, 14, 7, 7, 5] / [48, 45, 42, 39, 36, 32, 14, 21, 17, 14, 7, 7, 5]|
|PR_DF_AUTO_5bebe6918395_3b9c916b4f882df82b2d|PR_DF|6|2nd|適合|1,452,972,904.07|[64, 60, 56, 52, 47, 43, 38, 34, 30, 25, 21, 17, 12] / [64, 60, 56, 52, 47, 43, 38, 34, 30, 25, 21, 17, 12]|
|PR_PAULI_AUTO_8bf6eb88c735_7a94374fe4d2d0aaee49|PR_PAULI|4|2nd|適合|725,939,453.71|[48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 16, 13, 9] / [48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 16, 13, 9]|

重複除去：7件。詳細・固定SOURCE・26軸の価格対応はCATALOG.jsonを参照。
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
- `23160cc876807c0300be3de811f767b5810640a7:docs/research/d_pr_df_ld9_m6_price_20261116/C_HANDOFF.json`
- `7ccb3e63b687bd13426f3e53d96b93d918707641:docs/research/pr_df_ld7_pf_budget_q3_results_20261116/C_RESULT_JOIN.json`
- `1c4a83d1b841bfc23ba87c45f850fb4233f22c30:docs/research/pauli_ld5_two_vectors_q3_results_20261116/C_COMPARISON_JOIN.json`
- `c7c8c1edd0996ad9ef97629961f67ca508a182f0:docs/research/d_p7_y4_coarse_price_calibration_20261116/PRICE_AND_COST.json`
- `438d6c9a43351d45c31921a5676359dbc2c80b32:docs/research/h4_pf_optimizer_search_20261116/C_RESULT_JOIN.json`
- `5304fc149ac3e522f806457b36c03d91a170593e:docs/research/h4_optimizer_saved_results_intake_20261116/B2_SAVED_TOTAL_INDEX.json`
- `cbc471c913072de45c25a0ed22d51dff8a9b7034:docs/research/p7_s2_m12_one_results_20261116/C_RESULT_JOIN.json`
- `93ab9dbbe610d66eba39ea9299b070a13a867a3e:docs/research/p7_s2_m12_two_results_20261116/C_RESULT_JOIN.json`
- `93ab9dbbe610d66eba39ea9299b070a13a867a3e:docs/research/p7_s2_m12_two_results_20261116/SUMMARY.json`
- `b4296b110b129d226dbef2743c14f725a27c3bf6:docs/research/pf6_cost_budget_q3_results_20261115/C_RESULT_JOIN.json`
- `8ec7c882806809460d38a638a891bafcf18fafd1:docs/research/d_pr_df_ld_split_q3_resume_20261116/C_HANDOFF.json`
- `d7a976a629a6db68197b24f8f86734e109c8bf5d:docs/research/rpe_method_optimizer_h4_20261116/kappa_abs_round03/KAPPA_ABS_RESULTS.json`
- `d7a976a629a6db68197b24f8f86734e109c8bf5d:docs/research/rpe_method_optimizer_h4_20261116/kappa_abs_round03/best_points/LD5_kappa_x2_edd63e0befd3af591809/REQUEST.json`
- `d7a976a629a6db68197b24f8f86734e109c8bf5d:docs/research/rpe_method_optimizer_h4_20261116/kappa_abs_round03/best_points/LD5_kappa2_e3cb2cfda719117d6222/REQUEST.json`
- `1e107cf4de2186ee7df324412d18be9644a87d91:docs/research/d_pr_candidates_auto_20261116/run02/completion/comparisons/PR_DF/3b9c916b4f882df82b2d/C_RESULT_JOIN.json`
- `1e107cf4de2186ee7df324412d18be9644a87d91:docs/research/d_pr_candidates_auto_20261116/run02/completion/comparisons/PR_DF/3b9c916b4f882df82b2d/SUMMARY.json`
- `70578fdc21a6b4516a9f6b538ad9e305b8d7e28e:docs/research/d_pr_candidates_auto_20261116/run02/completion/comparisons/PR_PAULI/7a94374fe4d2d0aaee49/C_RESULT_JOIN.json`
- `70578fdc21a6b4516a9f6b538ad9e305b8d7e28e:docs/research/d_pr_candidates_auto_20261116/run02/completion/comparisons/PR_PAULI/7a94374fe4d2d0aaee49/SUMMARY.json`

PR-Pauli LD3 VPは保存Q3とAの26軸モデル価格を接続した条件付き参照総額。直接取得は別較正cellの2pathのみ、qualified_for_candidate=falseを保持する。

LD7_A_SHAPEの費用の95.9453590253993779917152%はq>64のm7〜m12へのモデル外挿（元SUMMARYの保存内訳）。qualified_for_candidate=false。価格モデル・配分・Q3・較正receiptの固定locatorはCATALOG.jsonに保持。

p7 Yoshida4・要求幅32h/5 A_SHAPE：直接16軸の部分額 56,433,222.00 RZ、モデル10軸の部分額 315,602,188.00 RZ。モデル価格全体の割合 84.831223%と、総額中のq>80未検証外挿の割合 75.400727%を区別する。価格誤差上界・CIは未取得。旧S2適合点とPR探索時の固定予算は変更しない。

LD9 V0の保存m6直接価格（別cohortの各軸4path平均、全13round費用はnull）：

|軸|LD9 RZ/shot|LD7参照 RZ/shot|差 LD9−LD7 RZ/shot|
|---|---:|---:|---:|
|X|492,573.00|393,837.25|98,735.75|
|Y|492,573.00|393,837.25|98,735.75|

再実行コマンドと追加方法は親directoryのHANDOFF.mdを参照。
