# B・D1・統合担当：Full-H κ2 m0再判定と固定配分会計

**受入2/2・保留0、各1062 RZ。固定配分のモデル参照T=1,246,382,808,920.55 RZ、m0部分額203904 RZ（約0.000016359661%）。両PF基準より高い。** 結果固定commitは `2f06a0a81b5edaecd9403e90d0a9f2f916e1b194`。

採択規則は481c4b7abの `FULL_H_KAPPA2_NORMAL_PATH_ZERO_ZZ_ADMISSION_PROPOSAL_V1`。coverage適用外と、非ZZ固定SOURCE/保存QPYによる未保存parityの代替証拠方式の**両方**を今回適用する。

対象はm0 FULL_H_2a9a68a11d08af234a8c/T.1/q1/R9、同一normal pathのX/Y、cohort D_FULL_H_KAPPA2_M0_REFERENCE_N1_20261115_V1/index0/seed15612141344395026302。元producer7f85f20e3、元receipt索引74cd9c262、証拠64624ea07を保持する。元parity戻り値は未保存のまま。旧D1 STOP・C426bの0/2/nullは変更しない。

固定SOURCEでrun01/RESULT.jsonに採否・元record・原案/証拠のlocatorを保存し、AXES.csvへ26軸を出力した。両軸受入の場合だけ、m0経験的単価＋F1保存24モデル価格を、B6c5569c3の1.5倍固定配分で既存accountへ渡す。PF比較基準はFull S2=615304502、p7 S2=435481090 RZ、差の符号はT−T_PF。

この再判定・費用比較は一構成・一配分の価格シナリオ。母E[C]/CI/正式G/winnerや、他配分の新Q3資格を与えない。取得・再fit・再予測は0。ロードマップ本体は更新せず、結果に応じた差分だけを追記して統合担当へ渡す。公開・remote照合後mandatory STOP。


## 結果固定参照

- 結果commit：`2f06a0a81b5edaecd9403e90d0a9f2f916e1b194`、実行SOURCE：`cbddabafc88bfa8d3ba04aed47a01868169cfdbe`。
- 機械可読正本：[RESULT.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2f06a0a81b5edaecd9403e90d0a9f2f916e1b194/docs/research/full_h_m0_zero_zz_admission_total_20261116/run01/RESULT.json)。軸別表：同commitの `run01/AXES.csv`、検査：`run01/CHECKS.json`。
- selector：`m0_admissions[0/1]` がX/Y再判定と元receipt、`rows[0..25]` がm/axis/physical_id/元configuration・total decoder ID/N/価格tier、`T`/`m0_partial`/`m0_fraction`/`m0_percent` がexact値。
- `comparisons[0].T_minus_T_PF` はFull PF S2、`comparisons[1]` はp7 S2。正の値ならFull-Hの参照総額が高い。正本CSV label/出典も各benchmark内に保持する。

受入2/2・保留0、同一normal path1、m0-X/Y各1062 RZ。m0部分額203904、T=1,246,382,808,920.55 RZ、m0割合約0.000016359661%。Full PF S2との差+1,245,767,504,418.55、p7 S2との差+1,245,947,327,830.55 RZ。

**m0は1 pathの軸別実測、残る24軸はF1の保存モデル参照価格。** 状態MODEL_SCENARIO_ONLY_WITH_EMPIRICAL_M0、正式E[C]/CI/G/winnerはnull。元parity/runtime markerはNOT_PERSISTED、今回根拠は独立対象0件・固定SOURCEによる代替証拠。採択は観測後であり、原記録は改変していない。

ロードマップ差分：m0経験的参照2/2受入と固定1.5倍配分の総額シナリオが完了、両PF基準より高い。F1原負値/全額null、D7係数・qualification、旧D1/C STOPはそのまま。新4人工test PASS、旧18証拠を再利用。新sampling/build/compile/re-simulation/RZ再計数/予測/fit/Q3=0。本体反映は統合担当、公開・remote確認後mandatory STOP。
