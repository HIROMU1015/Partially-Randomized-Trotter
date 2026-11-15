# C・Bへ：粗幅p7/Yoshida4 A_SHAPEの条件付き価格・費用

**条件付き総額372,035,410 RZ。** 直接16軸56,433,222 RZ＋モデル10軸315,602,188 RZ（84.831223%）。m9/q80は凍結予測とX/Yとも完全一致。ただし総額の75.400727%はq>80の未検証外挿で、価格誤差上界・CI・正式G/winnerはnull。追加科学実行はしない。

Cが読む固定commitとpath：

```text
c7c8c1edd0996ad9ef97629961f67ca508a182f0:
  docs/research/d_p7_y4_coarse_price_calibration_20261116/PRICE_AND_COST.json
```

同ファイルに元configuration・allocation ID、26 axes（m/axis/N/T/q/actual幅/physicalとcanonical ID/単価/N×単価）、直接/モデルの型、元receipt/元trial/旧SOURCE/cohort/frozen-model参照、軸別係数、m9のexact差、部分額・割合・3比較差・資源を収めた。直接値を優先し、条件付きモデル値をDIRECT証拠や厳密上下界へ昇格していない。

## 対象と固定結合

保存linear H4/1Å/STO-3G/tol-only DF10・B0I p7/標準Y4/PF要求幅32h/5/h_RPE=.1/P0除外。元configuration `dd8bfc248fdc67d42a6ae75df7eb0fbfcf34edc5e6ab3f3632e8784b47a7dbe0`。

allocation ID `B0I_P7_Y4_32H_OVER5_A_SHAPE_SAVED_INTEGERS_01`、X/Y各軸 `[37,34,32,29,27,24,22,19,17,14,11,9,6]`。Q3適合結果は `8855f420a9fe2577efc77f099c5a39ab0cb9b7e6:docs/research/d_p7_y4_coarse_a_shape_q3_20261116/C_RESULT_JOIN.json` を引用し、追加Q3は0。

価格quantityは1 shotの測定付きfull controlled Hadamard wrapper全体のcompiled RZ数。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、symmetric_directional、元native/lowering/fragment順、scalar c_H+tail identityをtotal Tへ一回、state preparation除外を保持。

## 係数・holdout・適用条件

m5/q5とm6/q10の各X/Yだけから、Cの既存exact fitで軸別に `C_a(q)=307+16539q RZ` を得た。early m0..m4の10軸は保存直接値で、実幅が異なるため較正から除外。m3は既存Cが受入れたlegacy native同等性証拠を保持し、元SOURCEを変更しない。

係数とm9予測は `d4ae4c408fb2218c041e2a24aa04c1c2d42df629:docs/research/d_p7_y4_coarse_price_calibration_20261116/FROZEN_MODEL.json` にm9開始前に保存。calibration_publicationは0e28adbd9580d37ae3f7a7063c8857477ccd7933。

| m9軸 | 予測RZ | 直接RZ | 観測−予測 | 符号付き相対差（観測分母） |
|---|---:|---:|---:|---:|
| X | 1,323,427 | 1,323,427 | 0 | 0/1 |
| Y | 1,323,427 | 1,323,427 | 0 | 0/1 |

2%規則等の新しい採否閾値は設定せず、結果はEXACT_MATCH_AT_Q80。m9をfitへ戻していない。m7/q20・m8/q40も未観測モデル参照であり、m10/q160・m11/q320・m12/q640はq80を越える未検証外挿。現幅閉包の移送や新QPY監査は0。モデル誤差上界/CIはUNKNOWN/null。

## 固定した3比較

差はすべて「粗幅A_SHAPE条件付き総額−比較先」。比較値は今回ユーザー指定の固定値で、後続別候補の結果へ途中更新していない。

| 比較先 | 比較先RZ | 差RZ |
|---|---:|---:|
| 同じA_SHAPE・現幅Y4 | 444,807,010 | −72,771,600 |
| 現幅p7/Y4最安適合点 | 435,641,568 | −63,606,158 |
| 指定のp7全体最安適合点 | 395,962,678 | −23,927,268 |

第1比較は同じN列での幅変更、後2比較は別配分/公式の固定基準との比較。モデル条件下では3件を下回るが、保証された達成費用・最適配分・winnerの認定ではない。物理的global RMSE・十分N・G_qualifiedも未認定。

## 元receiptへの戻り方

- early10軸：PRICE_AND_COST.axes[*].price_evidenceのsource/original_trial/proof_of_current_admission。元receipt cohortと元trial cohort（記録なしはnull）を区別。
- 較正4軸：`0e28adbd9580d37ae3f7a7063c8857477ccd7933:docs/research/d_p7_y4_coarse_price_calibration_20261116/calibration/run02/` のPRICES/SIDECARS/GUARD_CHECK/RESULT。科学SOURCE7f7622a94、raw e59329ddb。
- holdout2軸：`51c94af414e3171b5ee37cf6a8891afc858d67b3:docs/research/d_p7_y4_coarse_price_calibration_20261116/holdout/run01/` の同4ファイル。科学SOURCE/予測凍結d4ae4c408、raw0fb165703。

各新規軸のmeasured-wrapper fingerprint・RZ/CX/depth・compiler profile・元7作用gate、child/worker終了code/signal・資源・協調logは上記rawに保存。全6軸のC guard受入、SCI cache再利用0。cal2 workerで172.276 s、m9 1 workerで1,222.373 s。最大worker RSSはそれぞれ704,888,832／3,419,529,216 bytes。

初回metadata importの未展開referenceによる0-compile STOPと、calibration sidecarの旧namespace参照も保全した。元blob復元・参照結合のみを修正し、科学circuit/compile retry、gate緩和、m9 refitは0。実SCIの新規上限6件を守った。詳細REPORT/INITIALIZATION_RECOVERY/TESTS/RECOVERY_TESTS/BINDING_TESTS参照。

統合担当へのロードマップ差分：**粗幅Y4/A_SHAPEの幅固有較正とq80直接証拠、条件付き費用1点を追加。正式G/winnerおよびq640価格の誤差保証は未認定。** 本体は編集していない。旧SOURCE・原結果・STOPを保全し、公開照合後 **mandatory STOP**。追加compile・信号・Q3・他配分・価格fit・N探索へ自動進行しない。
