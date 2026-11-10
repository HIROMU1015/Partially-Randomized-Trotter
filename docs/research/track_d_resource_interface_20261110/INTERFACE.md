# 共通interface：科学条件とcontrol価格の分離

[実装](../../../src/trottertracks/track_d_resource_interface.py)／[REPORT](REPORT.md)。新builder、shot solver、model fitterは作らず、保存読取と既存集計への薄い接続だけを追加した。

## 型と責任

| 型／入口 | 内容 | 自動的には認定しないもの |
|---|---|---|
| ScientificInputs | Hamiltonian/DF/分割/PR/law/state/measurement binding、target、(m,q) horizon、law model、decoder | 元H誤差・正半径・alias・phase・IIDの真実性 |
| ControlProfile | compiler、cut policy、C1、topology、宣言phase規約 | 新compiler／cutでの作用・価格同等性 |
| MethodPrices | 上記両型、method ID、各round/axisの6価格・出典・幅、horizon資格 | 有限標本平均の母平均保証・長q外挿bias |
| SavedCDF8Study.common_shots | 固定B4/B6b条件に対する保存Nと出典 | actual N、別law／分割／targetでのN移送 |
| SavedCDF8Study.select | method/A/F/Sを選ぶ保存結果view | 新しいN・価格予測・総費用計算 |
| aggregate_supplied | 明示的なaccuracy、既存shot計画、MethodPricesを旧aggregate_totalsへ渡す | 上流資格・科学実行許可・成功確率の証明 |

ScientificInputsにcontrol methodや価格を入れない。ControlProfileにtarget精度やRPE Nを入れない。method IDは既存の `symmetric_directional` と `reference` のみ。Referenceはwhole-path一gadgetを指し、短いgadgetの繰返しへ置換しない。

## 安全な保存読取

```python
from trottertracks.track_d_resource_interface import SavedCDF8Study

study = SavedCDF8Study(repository_root)
strong = study.select(study.inputs, 'symmetric_directional', scenario='A')
reference = study.select(study.inputs, 'reference', scenario='A')
```

この例はGit固定blobを読み、旧resultのcolumnを選ぶだけである。N solver、model_value、price_records、aggregate_registered、aggregate_totals、cost providerは呼ばない。CLIも同じ読取経路：

```bash
python -B scripts/inspect_track_d_resource_interface.py --method reference --scenario A
```

出力には `shot_origin=SAVED_B4_OUTPUT_NOT_RECOMPUTED`、`total_origin=SAVED_RESULT_NOT_REAGGREGATED` と固定SOURCEが付く。NはB4の保存row、単価は8828b298のprices52.jsonからmethod別26row、総資源は同A/F/S result.jsonの保存column。実行SOURCE・B4 decoder・binding・horizon・旧STOPを照合する。

q1はcomplete event16、q≥2はcomplete step q/2。価格はfull-wrapperであり、Gaussian共有、head relative、tail sharing、step境界融合、scalar/identity、CSWAP、χ、readoutを含む。C1追加融合OFFの価格へ、別C1の節約量を加算しない。

## 既存集計への接続

`aggregate_supplied(inputs, accuracy, plans, prices)` は以下を要求する。

- accuracyのbinding/targetがScientificInputsと一致し、同じtargetのdyadic round計画と正整数shotsを持つ。既存のscalar `shots` はX/Y共通で、非対称Nへの拡張はしていない。
- 価格のmethod/科学binding/compiler/scope、全round×X/Y coverage、出典とwhole-interrogation単位が一致する。
- 6資源はfinite/nonnegativeで、widthをそのmean辞書へ混ぜない。有限sampleだけの価格は拒否し、conditional modelにはassumptionsを要求する。
- MethodPricesのhorizon資格は `qualified_horizon` 又は `explicit_conditional_horizon`。宣言の責任と出典は上流にあり、資格なしを自動的にTrueにしない。

adapterはその明示資格を旧集計器の `long_round_coverage` metadataへ橋渡しし、bindingを照合済みの共通target IDをcost rowへ付す。ΣNμの実装は **c1_stage3_accuracy.aggregate_totalsそのもの**を呼ぶ。widthだけは別に最大値を返す。条件付き会計であり、未知covariance/CI/model biasを0にしない。depthを実機時間に換算しない。

今回この数値入口を呼んだのは人工scalarだけ。H4の保存表はread-only経路のみ通している。将来、実入力を数値集計する場合は、その科学SOURCE/input/条件/実行範囲の許可が別途必要である。

## 不適格入力

保存domainと違うr/K/PF/LD/h/DF、state、law、target、decoder、horizonでは `ADDITIONAL_SCIENTIFIC_QUALIFICATION_REQUIRED`／`UNQUALIFIED_OR_UNSUPPORTED_CONTEXT` とし、N/価格/総資源はnull。compiler/cut/C1/phase/topologyの変更も保存価格・共通Nへfallbackしない。actual laneは追加証拠なしでは未資格。未知method/scenario、壊れた表、重複・欠測rowはエラーである。

返すdataclass／dictはcopyであり、呼出し側の変更で保存study内部や原本を変更しない。dataclassのfrozen指定自体を科学的なsealや承認証明と扱う仕組みではない。

## 別PR設定への拡張点

別設定には、新ScientificInputsと、元H誤差／phase／radius／IID／decoderを満たす十分N、両controlの測定同値性、同じcompiler/scopeでのmethod別価格と幅、そのhorizon資格が必要。新しい登録を行う際も旧固定CDF8 readerを変更して汎用モデルに見せない。旧A/F/Sを暗黙に移送しない。

元の短round optimizerやDFLevel5RCompiledCostProviderと型を無理に統一して実compileを起動しない。ここで整えた接続はmetadata/価格/総会計の境界であり、未知PR設定の科学証拠を自動取得するdriverではない。
