# D2向け：固定LD1残差2モデル・m4予測2件

**モデル/予測の固定commit：`3cac53da1a4a8b2228245593c74c1ff9ec1f5212`。** 正本path：

`docs/research/pr_df_ld1_residual_freeze_20261115/run01/FROZEN_MODELS.json`

科学実行SOURCE：`d83737b6d5241962ce39c81d39c56af3506c0cdd`。同SOURCEの`INPUTS.json`が採用24組の元S/C・SOURCE・receipt・seed/digestとm4構造値を固定する。入力保留0、X/Yともrank3、2fitとm4予測2件まで完了。m4価格開封・比較0。

## D2がそのまま読める項目

`models[axis]`（axis=`X`/`Y`）に、[1,q,qR]の順序、exact係数`coefficients.one/q/qR`、3較正cell平均、設計行列、rank/算術残差、`m4_prediction`がある。予測の比較正本は`m4_prediction.predicted_RZ.numerator/denominator`。`display_RZ`を使って再構成しない。D2は追加fit・元係数の再予測を行う必要がない。

```python
from fractions import Fraction
model = frozen['models'][axis]  # X/Yを個別に保持
value = model['m4_prediction']['predicted_RZ']
P = Fraction(value['numerator'], value['denominator'])
```

| 項目 | 固定値 |
|---|---|
| m / T / q / R | 4 / 1.6 / 16 / 32 |
| physical ID | `3df0cfd3356c358aeda8` |
| future cohort | `D_PR_DF_LD1_M4_HOLDOUT_NORMAL_20261115_V1` |
| paths | normal fresh index0,1,2,3、各軸4path |
| X予測 exact | `217164876899444503/1683627180032` |
| Y予測 exact | `217164876899444503/1683627180032` |
| 旧価格configuration | `587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911` |

元/total精度用configuration・canonical IDと元physical IDは予測の`identity`へ別々に保持した。新total decoder IDをreceiptの生成時IDへ付け替えない。

seedの固定参照：`2f423befbba5075bad4ede3057f89edd788d2661:docs/research/d_pr_df_ld1_m3_results_20261115/INPUTS.json#future_holdout.cell.seeds[index]`。各index16個のouter seed。`future_holdout.row/cell`と`holdout_source`を正本へ照合し、seed・候補を再生成しない。元`execution_authorized=false/prices_opened=false`は取得前planの履歴であり、今回Cのfit許可を証明する欄ではない。m4科学実行は今回の許可に含まれない。

## 採用入力から原receiptへ

- `input_source`は科学SOURCE d83737b6…の`…/INPUTS.json`を指す。
- `paths[*].input_selector`はINPUTSの24行へ、各行の`original_receipt/axis_receipt`等は元S/C・軸別fingerprint・cohort/index/seed/digestへたどれる。
- m0/m6はC c99d511e2受入のP2 fresh各4path。元role DEVELOPMENTを保持し、今回のモデル用途をCALIBRATIONとした。P1/replayは参照のまま。
- m3はD2 2592d442…のC_HANDOFFから、科学SOURCE2f423bef…・元結果/receipt4031dbab…へたどる。元scalar/ZZ/wrapper/compiler/profile・gateを照合し、full-wrapper statevector検証=falseは全入力で維持した。
- 6平均と24残差はFROZEN_MODELS、転記表は`CELL_MEANS.csv`と`RESIDUALS.csv`。3点3係数の一致は算術確認で、holdout成功を意味しない。

物理条件：保存H4 DF10/H/state/snapshot、LD1/S2/Δ.1/K2、S0 exactκ・fragment/basis順、scalar c+s_R総T一回、新ZZ、symmetric-directional control、測定付きfull wrapper/P0、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17。profile/scalar/preparationと原SOURCEはINPUTSの固定metadataと`source_facts`参照へ保持した。

## m4比較規則：取得前D2 planを無変更で継承

FROZEN_MODELSの`comparison_rule`が、D2実行SOURCE INPUTSの`model_plan.holdout_rule`と同一。後続の観測Oは同一軸の4compiled RZのexact算術平均。

1. P−O、`abs(P−O)`、`(P−O)/O`、`abs(P−O)/O`を報告する。分母はO。
2. O=0の相対誤差はnull、理由ZERO_OBSERVED_PRICE。
3. 欠測時はO/errorsをnullとし、4pathの定義を保持。3path平均や他軸の値で補完しない。
4. 許容差はnull、model verdictは`NOT_ASSESSED_NO_ADOPTED_TOLERANCE`。数値PASS/FAIL・PF2%・旧PR2%/95%を追加しない。取得gate/欠測の状態とモデル誤差を区別する。

## 価格資格と未解決事項

構造項はideal Iの保存E[S]、残差較正は元sampler Qの有限cohort。個別保存項・exact比・構造数値区間・law・kernel SOURCEは`m4_prediction.ideal_structural`と`residual_prediction`へ保持した。構造区間はtotal CIではない。

law距離、finite-cohort/係数不確かさ、rare、共有誤差、CI、適格全law E[C]はnull。`qualified_for_candidate=false`、statusは`CALIBRATED_UNVALIDATED_HOLDOUT_UNOPENED`。固定I/Q混合予測を、Q又はIの適格期待compiled価格へ昇格しない。Bの固定配分の条件付きQ3適合・十分N・価格資格は別であり、今回N/G/winnerはない。LD2・長roundへの移送もない。

ロードマップ差分：Cの指定3セル残差fit/予測待ちは解消し、D2の独立m4取得・記述的比較待ちへ進む。価格モデル資格・I/Q差・rare/長q-R/精度は未解消。ロードマップ本体は統合担当が更新する。

**mandatory STOP：Cは係数/予測の公開・remote照合と本HANDOFFで終了。m4 sampling/compile/開封、他round予測、CI/全law E[C]/N/G/winnerへ進まない。**
