# B2・D2：候補固有Pauli較正要求API

固定技術SOURCE：`7d6a8c385fc057de012c7e1e603bc0bc358f313c`。
branch `codex/a-pauli-on-demand-preparation-20261116`。
[REPORT](REPORT.md)／[SOURCE](SOURCE_INDEX.json)／[検収](ACCEPTANCE.json)。

## B2の接続点

1. Aの公開`7cf4e1f05`入口から、B2が選んだ1候補のprepared itemを受け取る。
2. 同じconfiguration/implementationに対するD2 `PriceRules.quote(policy=MODEL)`応答を受け取る。
3. 価格不足なら下記入口へこの2つをそのまま渡す。新しいL_D用のSPECSや係数を作る必要はない。

```bash
/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B \
  /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/a-pauli-on-demand-preparation-20261116/scripts/prepare_pauli_calibration.py \
  --preparation A_PREPARATION.json --price-result D2_APPLICABILITY.json --output CAL_PREPARATION.json
```

`--preparation`はA応答のitemまたは1候補のitems、`--price-result`は`PR_PRICE_APPLICABILITY_V1`の26軸応答。
ファイル、`COMMIT:PATH`、selector付きGit locator JSONを受ける。複数itemなら`--configuration-id`で選択できる。L_D・κ・幅等の再入力を利用者へ求めない。

ライブラリAPI：

```python
from trottertracks.pauli_calibration_request import prepare
result = prepare(root, prepared_item, supplied_d2_quote)
```

較正spec生成・driverのmetadata検査には、既存driverと同じ依存として、A worktreeの隣の`a-pauli-ld3-cal-source-13ab`にcleanな固定`13ab1acfaebe5f33958e1db67862f5c45725ab91` checkoutが必要。この較正metadata経路でもSOURCE確認を行う。取得環境を新設・変更する手順ではなく、既存専用checkoutを使用する。未取得なら不足SOURCEとして返す。complete価格のコピーだけなら較正driverは起動しない。

本入口はquote/fitを再実行しない。D2 outputはその生成requestの実装契約と対にして渡す。A側でも、A itemが要求する契約と固定generic recipeをreuse前に照合する。

## 応答と次の処理

| status | B2が扱うもの |
|---|---|
| `REUSE_SUPPLIED_PRICES_NO_CALIBRATION` | `reused_prices`のSOURCE付き26軸。新規較正なし |
| `REUSE_SAVED_CALIBRATION` | `saved_calibration_source`を再利用。新pathを投入しない。欠測価格の後段audit/登録をD2へ |
| `CALIBRATION_REQUEST_READY_NOT_EXECUTED` | `driver_request`と`.CALIBRATION_REQUEST.json`。B2が対象と科学範囲を選択する |
| `NEEDS_RULE_OR_IMPLEMENTATION_EXTENSION` | PF/K/actual step・signed recipe・compiler等の追加条件。既存S2へ置換しない |
| `NEEDS_IMPLEMENTATION_CONTRACT` | 要求されたlowering等と固定SOURCE recipeの不一致 |
| `NEEDS_INPUT_OR_FIXED_IMPLEMENTATION_SOURCE` | ID/schedule/axis/source欠測・混在。別L_Dを流用しない |

部分的に価格がある場合は`reused_prices`をそのまま保持し、`missing_prices`だけが次の対象になる。MODEL policyではfilled行をMODEL_REFERENCE、欠測行をNEEDS_PRICE_EVIDENCEと照合する。DIRECT cohortを代用しない。

保存cellのmetadataを調べるときだけ`--inspect-saved-cell`を指定できる。価格が揃っていても較正入力を再現するが、生成specはreuse-onlyであり、freeze/取得は拒否する。

## driver入力と固定実装

生成requestの形：

```json
{
  "schema": "A_PAULI_SPLIT_CAL_REQUEST_V1",
  "driver_spec": {
    "prefix": 3,
    "config_id": "dd5b881d308f6fc48d3c91d08cec22c5fbbac296f60242e9bfb433c3e0321847",
    "cell": {"cell_id":"Q8R660","q":8,"R":660,"T":0.8,"K":2},
    "reuse_only": true
  }
}
```

これは抜粋。完全specにはcanonical本体/locator、P0辞書・operators、m12対応、cohort/seed出所、出力先、実行branch、実装SOURCEを持つ。完全な保存・人工例は[examples](examples/)を参照。

```bash
python -B scripts/run_pauli_split_cal.py --request CAL_PREPARATION.CALIBRATION_REQUEST.json --metadata
```

この呼出しが実際のdriver入力検査。`--input-preflight`ではないためH4 restore/prepareを行わない。取得用JOBへも同じspecが渡り、workerはSPECS[prefix]へ戻らない。specとfrozen bindingのprefix/config/dictionary/cohort/cell/H/profile/operatorsが一致しなければH4処理前に拒否する。

実科学実行を別途承認した場合は、既存driverの入力preflight・freeze・commit/pushされたbinding・execute経路を使用する。今回の技術作業はその承認ではない。source・fresh output・一回取得・数値gate・STOPを維持し、新grantや承認証明書は作らない。

必要なら`--execution OPS.json`でcohort/base/private/branchを一括指定できる。これは運用設定であり、分割ごとのPF/λ/係数の設定ではない。未指定なら候補ID/cellから導く。保存cell再利用に別cohortを重ねない。

## D2へ渡すもの

- `implementation`：固定13abのexplicit-operators取得core、9c4006bfのPauli D2 lowering、元native/wrapper/providerの契約。
- `candidate_implementation_binding`：当該canonical ID・辞書identity・profile・P0 operator SOURCE。
- `driver_metadata`：既存cell ID、q/R/T/K、scalar一回会計、2 path×X/Y、16 step seed、最大4 compile。
- `post_acquisition_needs`と`requested_wrapper_times`：当該候補のnormal-body/境界分解、全必要Tのwrapper margin、D2登録/適用判定。

`implementation`が同じrecipeでも、新LDの数値検証や価格係数は取得済みにならない。qualificationは`SOURCE_BOUND_RECIPE_PENDING_NUMERICAL_INPUT_PREFLIGHT`。較正cellは`m=null`で、13roundのreceiptとして登録しない。

## 適用外と追加条件

現規則はS2/K2、actual step=binary64 0.1、元compiler/control/測定付きfull Hadamard wrapper。別公式/Kならsigned-piece・normal-body・境界の新規則/較正、別actual stepならsame-step較正またはSOURCE付きaudit拡張が必要。全必要Tのterminal-angle/nonzero-margin証拠もD2が扱う。Aは別公式へ切替えたり、他LDのA/H/Eを持ち込まない。

公開B2 `f01ce68ba`のPRICE_RULES.jsonは要約だけなので、raw 26行quoteとして入力しない。検収用のLD3/LD5価格fixtureは、公開PRICE_26AXISからexact値をコピーしたprojectionと明示してある。通常運用では実際のPriceRules応答を使う。

保存LD3/LD5は実JSONで入力再現。未登録LD4は人工SOURCE/dictionary readerによるdriver入力検査のみ。新H4 path/compile/fit/Q3は0。十分N・E[C]・CI・Gや候補順位の認定は行っていない。

公開後mandatory STOP。どの候補を次に較正するかはB2が決める。
