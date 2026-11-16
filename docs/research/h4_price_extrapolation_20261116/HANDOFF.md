# B2/C/D：新しい保存最安点に同じ価格依存分析を適用する

固定分析SOURCE：`080dbbffa1fbd1335b03e853cad4684db0cc340c`。
現在の対象はC catalog eb27138afのp7 Y4/bfc0c108…とPauli LD5/b36673ac…。
[REPORT](REPORT.md)、[exact 52軸](run01/AXES.csv)、[round感度](ROUND_SENSITIVITY.csv)、
[共有係数/反転条件](run01/ANALYSIS.json)、[測定案](MEASUREMENT_PLAN.md)。

## 既存成果の再現

```bash
/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B scripts/analyze_price_extrapolation.py \
  --inputs docs/research/h4_price_extrapolation_20261116/INPUTS.json \
  --output /tmp/h4_price_exposure
```

C Readerの固定Git読取とexact会計を再利用する。p7はその点の保存26単価、LD5は同じD2 `PriceRules.quote(policy=MODEL)`を再利用する。
LD5の元最安点で単価が永続化されていなかった事実を保持し、新しく保存したreplayを旧receiptと呼ばない。
原κの基準価格表をκ×2の価格表として貼り付けない。元のQ3適否・配分・予算を再判定しない。

## 新しい最安点を渡す入口

B2/Dが点を選択した後、その点自身のfixed configuration/配分/単価・SOURCEで
`--normalized <JSON>`へ渡す。候補選択、N最適化、価格モデルfitはこの入口の担当外。
実入力の例は[run01/NORMALIZED_INPUT.json](run01/NORMALIZED_INPUT.json)。
独立producerのnative physical IDとB canonical RMSE IDを混同せず、[対応表](IDENTITY_MAP.json)のようにframeを保存する。

最小構造：

```json
{
  "points": [
    {
      "point_id": "B2_SELECTED_FIXED_POINT",
      "configuration_id": "FIXED_CANONICAL_ID",
      "M": 12,
      "saved_total_RZ": {"numerator": 123, "denominator": 1},
      "allocation": [{"m": 0, "N_X": 1, "N_Y": 1}],
      "axes": [{
        "m": 0, "axis": "X", "N": 1,
        "configuration_id": "FIXED_CANONICAL_ID",
        "physical_id": "SOURCED_CANDIDATE_ID_WITH_FRAME",
        "q": 1, "R": 1,
        "unit_price_RZ": {"numerator": 10, "denominator": 1},
        "source": {"commit": "FIXED_SOURCE", "path": "prices.json", "selector": ["rows", 0]},
        "tier": "EXTRAPOLATION",
        "components": [{"id": "this_model.shared_slope", "value": 10, "feature": 1}]
      }]
    }
  ]
}
```

上記はfieldの抜粋であり、そのまま実行するfixtureではない。
実入力にはm0〜M・X/Yの全26軸、同じ配分のN、二つの点が必要。
出力にはexact総額、tier別費用、各軸/roundの単価ずれへの感度、共有係数のΣN×feature、固定配分の反転条件を返す。
全26価格が無ければ欠測として拒否し、0で埋めない。saved_totalとΣN Cが一致しない場合も拒否する。

- `DIRECT`は同じ物理cellの直接価格（有限path平均ならその資格を保持）。別R・実幅・cohortの較正をround directにしない。
- `SUPPORTED_MODEL_RANGE`は同候補・実装・実幅・lawでの較正/holdout範囲内の未測定参照。連続区間全域の保証を意味しない。
- `EXTRAPOLATION`は未測定q/R/角度等への適用。根拠のない範囲拡張をしない。
- evidence tierはcallerがSOURCE調査から明示する。helperはqの数値だけから科学的資格を推定しない。
- 同じparameterに同じcomponent idを使い、round間の相関を保つ。異なる候補/係数/軸を数値が同じだけで同一parameterにしない。
- モデル成分のSOURCEが欠けるときは、単価別感度までを使い、共有係数の分析は未取得と明記する。
- `DIRECT`行へモデルcomponentを付けて、既存測定まで係数変更で動かさない。
- exact有理数とbinary64を維持する。単価・N・candidate identity・actual step・compiler/profileの照合はproducer側でも保持する。

既存Nと価格が違う配分の一部を混ぜると誤るので、点のconfigurationだけで同一視しない。
今回LD5の選択配分はm4 N22。旧途中最安のN24、別の絶対κ2、旧基準κのR配置を使わない。

## 反転条件の意味

δCはactual−reference、ΔG=ΣNδC。価格誤差の確率分布を新設せず、
ΔG_p7−ΔG_LD5=38163827を固定二点の同額条件として扱う。
各点のQ3 statusや新最安探索結果は、外部の元評価器が管理する。
この分析はモデル価格の頑健性の診断であり、N/MSEの再評価・正式E[C]/G・winner認定ではない。
同じ費用誤差率でも、直接部分を固定する場合と全価格を動かす場合は反転境界が異なる。

## 次の担当と境界

Aからの追加価格確認の第一案はLD5 Q128R1、次いでp7 q160。両者とも現callerには小さなholdout metadata分岐が必要。
保存anchorと旧STOPを保持し、no-fit/fresh cohortの設計を先に用意する。
この作業ではDのcaller・B2探索規則・実行中run・Cの自動同期は変更していない。
実測を選ぶのはB2/D。取得直前の重複照合と別途の科学実行承認が必要で、今回の分析を取得承認と解釈しない。
