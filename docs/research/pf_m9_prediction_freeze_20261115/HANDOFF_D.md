# Dへの引渡し：m9の12予測・観測前固定

2026-11-15 JST。**12予測を受入、保留0。m9観測値は未取得・未開封・未判定。** 今回の利用者指示により、元係数からの12予測と下記m9限定規則を固定した。Dによる実回路取得・判定は後続作業である。

## 固定入力

- 予測結果commit：**`b1a7cd358daeab51ce538c747d0cdb8ad86c97d4`**。
- 正本：`docs/research/pf_m9_prediction_freeze_20261115/run01/PREDICTIONS.json`、`rows`の12行。
- 表：同`run01/PREDICTIONS.csv`。JSONのexact分子・分母と同じ値を転記している。
- 科学実行SOURCE：`375f026a30e3c1ed3179a73cbbe333578abff00f`、`scripts/freeze_pf_m9_predictions.py`。
- 元係数：`12372fa6b53c878dc5dd124e036572ba0cd82061:docs/research/pf_stage1_model_freeze_20261115/run01/FROZEN_MODELS.json`、`models[model_group_id].coefficients`。
- 物理点計画：`bcb67b2b2bb78776db1dd8d0625a3facb37cd298:docs/research/d_pf_allround_price_coverage_plan_20261115/NEXT_VALIDATION_PLAN.md`。
- C対応：`0b4b2d941cd669aaa45231e1236bc2d3158a9d12:docs/research/h4_five_method_candidate_alignment_20261115/ALIGNMENT.json`。対象行36/37、38/39、88/89、90/91、140/141、142/143。
- 事前規則：SOURCE `375f026…:docs/research/pf_m9_prediction_freeze_20261115/M9_POLICY.json`。予測正本にも同一内容を保持する。

上記commitは公開後も固定参照に用いる。予測結果は新しいFROZEN_MODELSではなく、元12モデルの**m9予測表**である。元係数・m6予測のreaderへこのJSONを代入しない。D側で必要なm9取得selectorは既存callerの最小拡張として扱い、Cでは実装・起動していない。

## wrapperの一意な対応

`(model_group_id, configuration_id, physical_id, axis, m)`をキーに、`q/T/actual_delta/actual_delta_binary64_hex`、route、formula、prefix、scalar、input_binding、split、fragment_order、signed_structureを照合する。各行に元モデル・係数・ALIGNMENTの固定locatorとselectorを保持した。

| formula | q | Full physical ID | p7 B0I physical ID | actual幅 |
|---|---:|---|---|---|
| S2 | 224 | a4ec53ff4e60d60a2930 | 1bd0a5dd4bbf876da1d6 | .2285714285714286 |
| Yoshida4 | 96 | b2c479e7805c4d1f0b30 | 9924c226ee246283ec54 | .5333333333333333 |
| M5 | 32 | a53547a5fb707b9550d5 | 60111109e0024d1a1064 | 1.6 |

全点m9、T=51.2、各X/Y個別。model_groupは`FULL_PF`又は`TRUNCATED_PF`、公式`2nd`/`4th`/`4th(m5_best)`、幅ラベル`16/7`/`16/3`/`16`、軸を`|`で結んだID。実幅は要求ラベルだけでなくbinary64値も照合する。

固定H4 DF10、Full10又はB0I p7、元H/state/snapshot・scalar一回・fragment順、P0、symmetric-directional controlled evolution、測定付きfull Hadamard wrapper。compiled全RZ数、Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、元native/lowering/control条件を維持する。元record由来のSOURCE/implementation証拠と同一条件を確認し、recipe名や価格の一致だけで受理しない。

## Dが後続作業で適用するm9限定規則

1. 上記物理点・軸・SOURCE・実装gateを満たす観測だけに適用する。
2. 保存した`predicted_RZ.numerator/denominator`をそのまま使用し、`abs(predicted_RZ-observed_RZ)/observed_RZ <= 1/50`を丸め前に比較する。Dが元係数を再fit・補正・予測し直す必要はない。
3. 有効な観測で条件成立はPASS、2%超はFAIL。取得・出所・実装gate不成立や欠測はUNKNOWN。分母が定義できない観測もUNKNOWN。
4. X/Y・方式・公式を個別判定し、**12軸すべてPASSの場合だけm9 campaign成功**。平均・poolでFAIL/UNKNOWNを消さない。
5. 閾値緩和・自動refit・別law/幅への置換・他roundへの資格拡張はしない。

現在の`observed_RZ`・`decision`・`campaign_pass`はnull、`qualified_for_candidate=false`。m9予測を保存したことは観測PASS・科学的な平均価格資格の付与ではない。規則の2%は統計的信頼区間ではない。

## 観測前の確認・保全

実行前metadata確認は[PREOPEN_CHECK.json](PREOPEN_CHECK.json)と予測正本`preopening`に保存した。公開直前の確認は[PUBLICATION_PRECHECK.json](PUBLICATION_PRECHECK.json)に保存する。公開ref・結果path名・関連job名・既知D namespace名と正確なlegacy target result pathの存在だけを確認し、観測内容を読んでいない。確認範囲の限界は同JSONに記録する。

m3保存12軸は元`3b130173…`の`SAVED_M3_DIRECT_OBSERVATION_NOT_CALIBRATION_OR_HOLDOUT`、m6の12軸は元`a558133…`の独立holdoutとして保持する。m3/m6値によるfit・補正・再判定は0。元較正24軸・係数・raw・receipt・cohort・STOPは変更していない。

ロードマップ差分は「m9予測/規則未固定」から「12予測・m9限定規則の事前固定済み、D取得・独立判定待ち」。直接価格実績48軸は増えていない。ロードマップ本体は統合担当が更新する。

**mandatory STOP：Cは公開・remote照合と本HANDOFFで終了。m9取得・開封・判定、他round予測、N/G、winnerへ進まない。**
