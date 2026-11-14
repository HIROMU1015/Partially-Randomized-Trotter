# D/Bへの実行前HANDOFF

## 保存M5/m4：受入PASS、再compile不要

Full PF p10 M5、actual16h、m4/q1、physical `0cd1527c80f7b9841a83`。X/Y各85960 RZは保存値のcopy。原`EXPLORATION/index0`、scalar c、native・wrapper・compiler・入力bindingが一致した。詳細は[REPORT](REPORT.md)、固定bundle `3368805575d459d948e3afe7b1e54b4814f64129`。

- 受入record：[PRICES.json](PRICES.json)。現Dの`price_bindings`を再利用。SOURCEには原carrierとtrial locator/hashを保全。
- 使用可能なsidecar：[SIDECARS.json](SIDECARS.json)。`IMPLEMENTATION_EVIDENCE.json`をbundle commitに固定済み。[SIDECARS_UNBOUND.json](SIDECARS_UNBOUND.json)はplaceholderを残す工程記録なので消費しない。
- 現C628 guardで両軸を実検査済み：[GUARD_CHECK.json](GUARD_CHECK.json)。probe policyは**技術的互換性確認専用**で、科学的な比較policyとして流用しない。
- B eecはC628のguardをpinしている。decoder由来の新canonical configurationへ移す場合はBの公開`rpe_total_binding.bind_prices/check_prices`を使い、旧identity・元SOURCE・assertionを保持する。文字列のID付替えやreceipt再発行は不要・不可。

## Dの取得対象：34新規wrapper上限を維持

|方式|較正の新規wrapper|既存較正receipt|独立holdout wrapper|新規合計|
|---|---:|---:|---:|---:|
|Full PF p10|10|M5/m4 X/Yの2|6|16|
|Truncated p7 B0I|12|0|6|18|
|計|22|2|12|34|

[CELL_PLAN.json](CELL_PLAN.json)のCALIBRATION/HOLDOUTを使い、3公式×2軸を維持する。全36行のうち2行は保存受入、34行だけが新取得。B0/B0I比較の追加2件、p8、直接価格8 pilot、旧巨大T2、他方式を追加しない。

**現状の実行判定**：

- `62ae1bd5`の分類・scalar・physical ID・record/sidecar仕様とC guardは整合済み。既存のnative/compile/cache/receipt/gateをそのまま使用できる。
- **ただし、未承認のscienceを開始できる状態とは別である。** `62ae1bd5`の両`execute()`は全CALIBRATION/HOLDOUTセルを一括取得し、全取得後に入力を出力する。既存Cの`NEXT_PRICE_VALIDATION.md`にあるfit/freeze後のholdout取得順を満たすstage選択がない。
- Dへ依頼する必要最小差分：既存セル取得処理にCALIBRATION/HOLDOUTの段階選択を加え、前段結果を上書きせず、元cell/axis/seed/compiler/gate/STOPを変えない。較正後にCのmodel freezeを受け、次の段階で残りholdoutのみ取得する。C側ではD SOURCEを変更していない。
- これを解消し、ユーザーが許容差・比較policy・有限科学scopeを採択/承認したら実行可能。固定`62ae1bd5`のCLIを現時点でそのまま実行する指示は出さない。SOURCEの技術更新が必要である。

## 承認後のC処理（未実行）

1. 価格取得前に[POLICY_PROPOSAL](POLICY_PROPOSAL.md)から許容差・分母・比較policyをユーザーが採択。feature `one,q`、12個のmodel_group、m4/m5較正・m6 holdoutを固定する。
2. Dは較正22新wrapperだけ取得し、保存2 receiptとともに公開。元gates、scalar、保持block、physical ID、full measured wrapperの不一致なら部分保存STOP。retry、tolerance変更、片軸への切替はしない。
3. Dの出力sidecarにある`FILL_WITH_PUBLISHED_COMMIT`を**実際に公開された固定assertion commit**へ結合する（価格を変更せず、SOURCEの虚偽付替えもしない）。C guardを通す。placeholderのままGへ渡さない。
4. 各`(route, formula, width_over_h, axis)`について、元physical候補・SOURCE・profileを照合。D `fit_inputs`→C `price_model_validation.record`を使用する。Full PFの保存M5/m4は[SAVED_CALIBRATION_INPUTS.json](SAVED_CALIBRATION_INPUTS.json)と結合し、同cohortへのpoolや平均を行わない。
5. m4/m5の2点だけで`fit(features=['one','q'])`をexactに実行し、係数・design/source・許容差・役割を固定commitへ保存する。m6をfit入力へ含めない。
6. Dは残り12 holdout wrapperだけ取得・公開。C guard後、同groupのm6へ`validate`を一度適用する。holdoutで不合格なら係数・feature・閾値を変えずSTOPする。成功してもmodel資格は自動昇格しない。
7. fit/holdout結果を公開してSTOP。長q外挿、G_paper、正式G、candidate順位へ自動進行しない。

`fit_inputs`は呼出側が正しい公式・幅・軸を選ぶことを前提としている。`price_model_validation._domain`に公式/幅の独立fieldはないため、異なるmodel_groupをまとめてfitすることは禁止する。X/Y別入力に同じ`cell_id`文字列があっても、一つのmodelで混ぜなければ問題ない。

今回の承認は保存receiptの技術受入・SOURCE照合・人工/回帰テストまで。新compile・実fit・予測/残差検証・Gに対する科学実行承認はない。**mandatory STOP**。
