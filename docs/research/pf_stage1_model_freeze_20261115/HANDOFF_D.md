# Dへの引渡し：12個別PF価格モデルとm6予測の凍結

**状態：較正済み・係数とm6予測を凍結・独立holdout未評価。** 12/12 groupを固定Stage1の各m4/m5から個別fitした。保留groupは0。m6観測価格は取得・開封・判定していない。G/N/winnerはnull。

## 固定取得先

Repository：`HIROMU1015/Partially-Randomized-Trotter`

C branch：`codex/m5-m4-receipt-admission-20261114`

**凍結結果commit：`12372fa6b53c878dc5dd124e036572ba0cd82061`**

- [凍結正本 FROZEN_MODELS.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/12372fa6b53c878dc5dd124e036572ba0cd82061/docs/research/pf_stage1_model_freeze_20261115/run01/FROZEN_MODELS.json)
- [12行のFROZEN_SUMMARY.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/12372fa6b53c878dc5dd124e036572ba0cd82061/docs/research/pf_stage1_model_freeze_20261115/run01/FROZEN_SUMMARY.csv)
- [REPORT](REPORT.md)、[D_CONSUMER_CHECK.json](D_CONSUMER_CHECK.json)、[INPUT_AND_EXECUTION_CHECKS](run01/INPUT_AND_EXECUTION_CHECKS.json)

凍結正本のSHA-256：`845c705082565a2b44d198bddb3034497620d991a00604454394ae5acda9651a`。以後の文書commitでもこの正本を変更しない。実行SOURCEは`79c951b38bf9d8f1c80f28b2e7f82846d85c67e5`。

## 既存D readerへの入力

```python
frozen = {
    "commit": "12372fa6b53c878dc5dd124e036572ba0cd82061",
    "path": "docs/research/pf_stage1_model_freeze_20261115/run01/FROZEN_MODELS.json",
}
# D 91365dcf の既存API。取得/compile/判定を起動する関数ではない。
accepted = pf_price_stages.frozen_models(root, frozen, calibration_inputs)
```

`calibration_inputs`には既存D Stage1の各routeのC_INPUTSをそのまま渡す。12 group一括・各route6 groupの両方で、**固定D `_canonical/frozen_models`本体を無変更で読取確認しPASS**した。技術確認中はfit/predict/holdout validateを禁止し、原SOURCEを参照した。Dのstage runner・compile処理をCへ取り込んでいない。

Stage1入力は`b190bd815c886446291fac8b281c3c7df7343372:docs/research/d_pf_stage1_calibration_20261115/C_HANDOFF_INDEX.json`。各`C_inputs_source`の`[s['record'] for s in saved_records] + new_records`だけが較正データである。

## groupとidentityの照合

model_group IDは `route|formula|width_over_h|axis`。routeはFULL_PF/TRUNCATED_PF、formulaは`2nd`/`4th`/`4th(m5_best)`、widthは`16/7`/`16/3`/`16`、axisはX/Y。計12個、JSONの`models[group_id]`で一意に選択する。

各modelの`calibration`は既存V.fitの元cell_id/source。`input_locators`に2入力のC_INPUTS selector、元price、guard、sidecar、cohort、physical候補、q/実幅、scalarの量と値・一回適用、H/state/snapshot、split/fragment順、compiler profileを保存した。

保存4件は両routeのM5 m4 X/Y。元EXPLORATION・155529e20原価格・trial/hash/receiptを維持した。新20件のcohortとも他axisともpoolしていない。

`m6_prediction`に固定CELL_PLAN上のconfiguration ID、physical candidate ID、q、axis、feature、`predicted_RZ`（numerator/denominator）がある。m6 physical IDはFull S2 `4f3984c6d381fa0ca848`、Y4 `7727a3fec25e033629f1`、M5 `8db0bd3c1a5e1eeb541f`、p7 S2 `fb512f2d5762df64bfae`、Y4 `5e19b5ded7386572684f`、M5 `4bc967fe8dbc6f814146`。各X/Yを独立に照合する。

元scalar、full measured controlled wrapper、P0、native/lowering、Qiskit1.3.0、rz/sx/x/cx、opt1、seed17が異なるreceiptを同価格として受け入れない。C側の再compileやD原guardの緩和はしていない。

## Dの別作業で適用する判定規則

policyは`91365dcf0610362a96aab101fbe0ec0e78f58af1:scripts/pf_price_stages.py:COMPARISON_POLICY`と完全一致し、凍結正本にも保存した。

**各モデル：`abs(predicted_RZ - observed_RZ) / observed_RZ <= Fraction(1, 50)`。campaign成功：12件すべてが個別PASS。** 観測RZを分母に用いる。係数・予測はJSONのexact値を使い、丸め表示・route/axis平均を比較に使わない。2較正点の残差0をholdout PASSとして扱わない。

観測m6と不一致なら元条件のまま結果・理由を保存してSTOP。自動refit、特徴量追加、閾値緩和、他route/axisの価格による補完を行わない。不明SOURCE・scalar/実装条件差がある場合も保留する。

現時点は全モデル`CALIBRATED_UNVALIDATED`、`qualified_for_candidate=false`。m6合格だけでm7〜m12、別幅、別compiler/loweringへ外挿資格を与えない。モデルsystematic・共有共分散はnull。正式な全RPE価格tierやG/winnerは未認定。

## 残る作業・ロードマップ

入力未解決0、モデル凍結待ちは解消。次は**Dの別実行指示**によるm6 12wrapperの取得・開封・個別判定である。今回のC許可はDの科学実行許可を代行しない。長qの移送証拠、精度側の十分N、全G比較は別課題。統合担当はロードマップの決定論PF価格を「12モデル較正・凍結済み、m6独立holdout待ち」へ更新する。

**mandatory STOP。Cはここで終了し、holdout取得・開封・判定、他round予測、refit、Full-H/Pauli、MSE/N/Gへ進まない。**
