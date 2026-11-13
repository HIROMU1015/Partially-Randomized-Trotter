# 最新Full PF3候補：次の有限価格検証案（未承認・未実行）

元native条件を使うか、新lowering/native scenarioを選ぶかはユーザー判断。既存directはD e7の元native/full measured XY/P0限定。保存価格の数値受入、fit、予測/残差比較、新compileは今回実行しない。

## 最新domainと役割

要求幅はA71f04be7のexact h相対分数を使う。S2/Y4の保存m3はそれぞれactual2h/4hであり、新定常幅16h/7・16h/3の校正に使わない。M5 m3もactual8hで別domain、m4のみ16hの保存校正点になる。3候補をpiece数で除外しない。

まず新しい定常幅の直接anchorだけが必要なら、S2 m4/q7とY4 m4/q3の2round×XY＝最大4 wrapperが最小の初期取得案。M5 m4/q1は同native保存receiptの受入を優先する。これは同T1.6の一部分比較であり、全RPEの価格区間/G順位を確定しない。

formula/actual幅ごと `a+bq` を識別して独立holdoutを置く場合は次の一括有限案になる。

|候補・固定actual幅|校正|凍結後の独立holdout|新round/最大wrapper|
|---|---|---|---:|
|S2 16h/7|m4/q7 `b7c4e73538ec19070cde`、m5/q14 `8df73ab164fac57bfd6c`|m6/q28 `4f3984c6d381fa0ca848`|3 / 6|
|Y4 16h/3|m4/q3 `3cf8eddfb3b78e96f5d2`、m5/q6 `72383d110445e479155d`|m6/q12 `7727a3fec25e033629f1`|3 / 6|
|M5 16h|保存m4/q1 `0cd1527c80f7b9841a83`、新m5/q2 `c63a835541462a1c264b`|m6/q4 `8db0bd3c1a5e1eeb541f`|2 / 4|

合計新8round・最大16 full measured wrapper。上の2round/4 wrapper初期案はそのsubsetであり加算しない。旧候補の4round/8 wrapper案は別計画のまま保全する。Full PFは決定論価格で、PR trajectory cohort/samplingは0、両軸は個別取得する。X/Y片軸compileは採択しない。元seed_transpiler17、SOURCE/fragment/signed-time/scalar一回・compiler・wrapper条件を固定し、既存D native取得/receiptを使う。

科学scopeを承認する場合、保存M5 m4の両軸受入、5新校正round/10 wrapper、3公式×2軸のfit/freeze、3 holdout round/6 wrapperとその予測/残差比較を明記する。保存M5 m4が受入不能なら自動取得し直さずSTOPして判断へ戻る。別native scenarioには件数をそのまま移さず、保存点の同値性を再確認する。

## 初期検証後も未資格の部分

- early actual幅は別domain。保存directの4round以外は不足で、定常幅modelをm0〜m3へ無条件適用しない。
- 2校正点＋一つのholdoutは使用q全域の保証にならない。m7〜m12、最大q1792/768/256への移送にはSOURCEによる閉包・有効な誤差囲い又は追加の必要域検証が必要。m12巨大compileを最初から要求しない。
- 旧PR C1、H6 Affine、rank12のpartial PF、C/Sの観測比を新Full PF全roundの係数/厳密界にしない。
- 保存directは固定条件で決定論的な価格。q-modelの系統誤差をn=1の標本SEや0で埋めない。XY/round/公式の共有誤差を無根拠に独立としない。

## 必要精度と停止条件

BのPR-LD1/LD2 DPは最新Full PFの十分Nではない。候補別の適格N又はN集合、比較margin/regret、対象price law/主張水準が未確定なので、価格精度・正式GはUNKNOWN。全roundに同時に有効なprice誤差b_iがある場合だけ固定非負Nに対して `|Ghat−G|≤Σ N_i b_i`。根拠付き区間のG差が識別できる精度を目標にし、今回は数値評価しない。

SOURCE/scalar/profile/作用gate不一致、欠測、精度未達では部分結果を保存してSTOP。閾値緩和、別幅/別law、再fit、候補除外、compile追加へ自動進行しない。ユーザーが比較scenarioと上記有限scopeを判断し、許可後にD/C/Bが各担当範囲を実行する。今回の公開後はmandatory STOP。
