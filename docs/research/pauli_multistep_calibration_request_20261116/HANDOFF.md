# B2・D2：実step別の較正要求

旧`dd81b846f`から、`pauli_calibration_request.prepare`、`prepare_pauli_calibration.py`、`run_pauli_split_cal.py`を拡張した。canonical configurationとD2のraw 26軸MODEL応答を渡す。B2の要約PRICE_RULESや丸め表示から価格を作らない。

```python
from trottertracks.pauli_calibration_request import prepare, prepare_many
result = prepare(root, canonical_configuration, d2_applicability)
batch = prepare_many(root, [
    {'configuration': config_a, 'quote': quote_a},
    {'preparation': prepared_b, 'quote': quote_b},
])
```

従来のA prepared itemもそのまま使える。直接canonicalを渡すときはbodyを保持し、q/実step/Rを再計算しない。元locatorを残す場合は`canonical_preparation(config, source)`を使う。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B \
scripts/prepare_pauli_calibration.py \
  --preparation CONFIGURATION.json --price-result D2_APPLICABILITY.json --output PREPARATION.json

# 複数候補の同じ較正cellを1件にする。listの各要素は上記APIの入力。
python -B scripts/prepare_pauli_calibration.py --batch SELECTED_INPUTS.json --output BATCH.json
```

`calibration_requests[*].driver_request_file`と`driver_validation_command`で各requestを実driverのmetadata入口へ渡せる。1要求なら旧`driver_request`/`.CALIBRATION_REQUEST.json`も保持する。複数なら`.CALIBRATION_00.json`等を別々に生成する。旧`--configuration-id`による1候補選択も維持した。

|返却状態／action|B2の次処理|
|---|---|
|REUSE_SUPPLIED_PRICES_NO_CALIBRATION|SOURCE付きfilled価格を返す。新規較正0|
|REUSE_SAVED_CALIBRATION|元BINDINGをD2のaudit/登録へ。再取得しない|
|CALIBRATION_REQUEST_READY_NOT_EXECUTED|1cellの科学scopeを選ぶ。自動実行しない|
|CALIBRATION_GROUPS_READY_NOT_EXECUTED|`calibration_requests`のunique cellを選択。同stepの複数roundを重複投入しない|
|NEEDS_RULE_OR_IMPLEMENTATION_EXTENSION|`round_actions[*].needs`を解消。使える保存軸・他groupは保全|
|NEEDS_INPUT_OR_FIXED_IMPLEMENTATION_SOURCE|SOURCE・identity・出力先等の不足。別候補で補完しない|

`next_action`は各groupにもある。`execution_authorized=false`であり、この返却自体は科学実行の許可を意味しない。

## 適用条件

- 固定H4 DF10のH/state/snapshot、h/M、P0、scalar一回、S2/K2、元辞書・fragment順、Pauli D2 lowering、compiler/control/full measured wrapperを保持する。
- stepはcanonicalのexact T/qとbinary64を厳密照合する。非binary64-exactなstepは不足SOURCEとして返す。厳密1/5とbinary64 .2を丸めて同一視しない。
- cellはq8、Rは当該step groupのcanonical R最大値、Tは8×実step。m12 fieldは全scheduleの来歴であり、他step groupの較正anchorではない。
- 複数roundへのdomainは「この較正と後続audit/モデル検証を必要とする対象」のmetadata。較正のみで全q/R/Tへのモデル資格やwrapper保証を与えない。
- 複数candidate間の重複除去は同じ物理request・law・cohortに限定。canonical ID・requestconsumer・元価格SOURCEは別々に残す。
- 実stepが違えば、短いcell名が同じでも出力base/privateは別。explicit overrideで衝突させると拒否する。

現幅の保存LD3/LD5再現、別幅のround分類、batch重複除去、実CLI→driver metadata→worker入力の人工15件PASS。[検収JSON](ACCEPTANCE.json)にcaseごとのround/action/domainを保存。新H4 signal/path/compile/fit/quote/Q3は0。旧固定成果・STOPを保持し、実科学取得へ進まない。
