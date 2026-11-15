# C向け：Full-H κ2 m0の固定normal path取得結果（単価null）

**mandatory STOP。C_0X=C_0Y=null、両軸とも受入保留。** 指定normal pathを一度生成し、X/Yを各一回compileした。
元観測RZはX/Y各1,062、準備照合と7数値gateはPASS。ただし必須のZZ置換実績条件が不成立だった。
観測値1,062を参照単価として使わず、nullと区別する。再抽選・seed変更・retry・追加pathなし。

| 項目 | 固定値 |
|---|---|
| 規則 | FULL_H_KAPPA2_M0_NORMAL_PATH_REFERENCE_V1（今回の比較用に採択済み） |
| physical | FULL_H_2a9a68a11d08af234a8c、m0/T.1/q1/R9/K2/κ2/P0 |
| cohort | D_FULL_H_KAPPA2_M0_REFERENCE_N1_20261115_V1 |
| path / seed | index0 / 15612141344395026302、独立path数1 |
| 科学SOURCE | 7f85f20e38af4df1a649ac239e57bfe76d0899ee |
| 固定結果SHA | b25d7da6dedebc4cd22e8d6f00e6de75d7072292 |
| 観測X / 参照C_0X | 1062 / null |
| 観測Y / 参照C_0Y | 1062 / null |
| 件数 | sampling1、compile2、価格再利用0、受入0/保留2、receipt欠測0 |
| 96 shot | 単価へ乗算していない |

停止理由：保存9事象はorder0、central supportがすべて1 qubitで、ZZ置換の機会が0。
one-body適用は6件ある。固定P1.pilotの総合条件はparity.replaced_controlled_zz > 0を要求する。
元7数値gateの不一致ではなく、このcoverage条件の不成立である。
individual parity/scalar-event/XYの戻り値は例外時に未保存であり、PASSへ補完しない。
[STOP_DETAIL](STOP_DETAIL.json)と[REPORT](REPORT.md)を参照。

各軸は別measurement/fingerprint/元compiler記録を持ち、同値を理由に一方をコピーしていない。
[HANDOFF_C.json](HANDOFF_C.json)のaxis_rowsから、原観測値、null価格record、元compile JSON、wrapper/compiled QPY、
native数値gate/event/cohort/seed、停止理由へ固定SHA/path/selectorで辿れる。
公開SHAのbytes/hash・来歴照合は[PUBLICATION_CHECK.json](PUBLICATION_CHECK.json)。
元QPYは測定末尾だけを読んで軸を帰属し、再simulation・RZ再計数・build/compileなし。

D7 preparation hash923253…、H/state/snapshot、fragment/component順、scalar総T一回、新ZZ lowering、
Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、測定付きfull controlled wrapper/P0を確認した。
価格の意味は採択規則が予定する1 pathの経験的参照までで、母期待値/CIの資格は付けない。
現在はrequired gate未受入のため、その経験的参照単価自体もnullである。C科学的受入はNOT_PERFORMED。

ロードマップ差分：規則採択と有限取得は実施済み、元2軸compile証拠を保全、ZZ coverage条件の扱いの判断が残る。
元F1の負値/全額null、D7資格、旧STOPを保全。D1はgate修正/再資格化/再取得、96倍、m1〜m12との新シナリオ総額、
fit/Q3/E[C]/CI/N/G/winnerへ進まない。ロードマップ本体は統合担当へ。commit/push・remote照合後mandatory STOP。
