# D2/B2：新P0 builderとLD7–9の通常入口

辞書の科学SOURCEは **`25a08afafa32af2ca52bca4f63233c38b3af0e6e`**、公開metadata API/registryは **`9f00ddd86a44b5d1443f29b5f02dbf1db4ef9ab8`**。
実行branchは `codex/a-pauli-p0-no-drop-preparation-20261116`。
固定locatorの機械可読正本は
[`h4_no_drop_splits.json`](../../../configs/pauli_candidate_preparation/h4_no_drop_splits.json)。
今回の完了範囲は辞書/P0と基準configuration/13round scheduleまで。信号・価格・Q3は未取得。

## 通常の要求と呼出し（保存metadataのみ）

B2のrequest形式は変更しない。例：

```json
{"kind":"pauli_native","items":[{"item_id":"B2_SELECTED_BASE","configuration":{"L_D":7}}]}
```

上記を既存REQUEST_LD7.jsonで渡せる。更新済みの専用worktreeから：

```bash
/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B scripts/prepare_pauli_candidate.py \
  --request docs/research/pauli_p0_no_drop_preparation_20261116/REQUEST_LD7.json \
  --output /tmp/pauli_ld7_prepared.json \
  --p0-builder no-drop
```

LD8/9はrequestファイルだけを置き換える。cacheも`--prepare-missing`も不要。
default（builder指定なし）でも登録された新辞書が見つかる。
応答は`status=GENERATED`、`preparation_status=REUSED_CONFIGURATION`で、
`configuration_source`は下記固定commit、`dictionary_builder_id=P0_NO_INTERMEDIATE_DROP_V1`となる。
`signal_plan.request_rounds=[0,...,12]`、`readiness`は構成true／信号false／価格false。

## 固定locator

全configurationはcommit `c034dfa024e7134077001d23ca08bb60a424877e` の
`docs/research/pauli_p0_no_drop_preparation_20261116/LD{7,8,9}/PREPARATION.0.CONFIGURATION.json`。

|7|`925e613adacd4fef37623e0ec426979367dc65d5aaa91e717b92503c9a75272d`|`7ba6fe757a3decb0eee3b9147100e143841ad9c9`|
|8|`9766ce5ec41202db0aab8bb083822cce779c69d522f22e38357102453697ff6e`|`c96bca4d0af0c4db8e213a2c65488f1a2e9e4d47`|
|9|`7a73dbd84662ce0387c044f50a3d4bde0f1939f9ce98cef04b80dab85f5e522e`|`f5bf35c9033abc645c34f23e4ec79bee15a5f3f2`|

辞書は各行のP0 commitの
`docs/research/pauli_p0_no_drop_preparation_20261116/p0/P0_NO_INTERMEDIATE_DROP_V1/LD{n}/LD{n}_DICTIONARY.json`、
同じディレクトリの`result.json`が全P0根拠、`DESCRIPTOR.json:descriptor`が保存DF descriptor。
[COMPARISON.json](COMPARISON.json)とregistryに全SHA・全path・各round physical IDを保存している。

## D2への接続箇所

確認したD2 `69ef01cdc` の `Completer.pauli_configuration` は既存A CLIを使う。
D2が次に選択する候補の `settings.pauli_workspace` を、上記API/registryを含むcheckoutへ向ける。
A_BASEが保存configurationを返すため、旧LD7–9の`NEEDS_SPLIT_PREPARATION`に対するP0再実行分岐へ入らない。
D2のrequest例とA_BASE→A_DESIGNの受入記録は[reuse_d2/](reuse_d2/CHECKS.json)。
公式/Δ/κ変更は、取得した同じ辞書から既存B2 mapperへ渡す。辞書は作り直さない。

旧D2 runのHELD/STOPやA_BASEのRETURNを再開・上書きしない。
既に停止記録がある設計は、新configurationの固定locatorを`request.configuration`に指定するなど、
新しい物理identityを持つ選択としてD2側の規則で扱う。AからD2のstate/探索規則は変更していない。
本HANDOFFは旧STOPの解除や後段科学実行の承認ではない。

## 新旧成果・後段入力

- builder25a08afafと旧OpenFermion-default辞書を区別する。payloadにbuilder ID/SOURCEを含め、hashが変わる。
- cacheの旧`splits`は保存し、新`split_versions[prefix][builder_id]`を併存する。
  明示`no-drop`は旧辞書へfallbackしない。同policyで違う固定辞書との競合は理由を返して拒否する。
- 新辞書には各360非identity実係数があり、λ_PはLD7/8/9で
  0.00042859681055870485／0.00013779703426245883／0.0000005473824345352291。
  DF descriptorのλ_RをPauli mapperへ代入しない。
- `configuration`と`schedule`、`dictionary_identity`、`descriptor`をそのまま信号入口へ渡す。
  LD8は返却`signal_plan_command`の`--tail-descriptor`も保持する。
- `price_handoff`はD2 PriceRulesの未実行request。別LD/旧hashの係数や価格を転記しない。
  candidate固有の実装契約・辞書hash・law・SOURCE・実step幅を後段で照合する。
- 診断shadowの残差一致と、今回の通常入口/最終辞書のP0合格は別証拠。
  P0資格だけでN/RMSE/Q3/価格をPASSにしない。

## 今回の実行とSTOP

実P0の呼出しは通常入口の `--prepare-missing --source 25a08afafa32af2ca52bca4f63233c38b3af0e6e --p0-directory ... --p0-builder no-drop`。
LD7/8/9各1回を実行済みである。fresh科学実行の再承認なしにこの実行commandを再使用しない。
SOURCE25a08afafからの数値builderは不変で、9f00ddd86は登録・locator優先のmetadata修正。
保存再利用15候補解決は全てP0/信号leafを禁止して確認した。
信号・価格・Q3を起動せず、公開照合後mandatory STOP。
