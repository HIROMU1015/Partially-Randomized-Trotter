# B2・Aへ：共通Pauli入口の新旧版受入

- B実装：`830cf9bd87ec5b1d27d760b1762d60cc828abeac`。
- 入口：`src/trottertracks/rpe_pauli_signal_intake.py` : `accept_saved(root, integration=..., handoff_source={commit, path})`。既存の[呼出し例](../pauli_common_entry_intake_20261116/HANDOFF.md)と同じ。
- 新A組立て：`4e5b8bef634e3f071fe63df9683d7245df984d98`。
- 旧A組立て：`93a5ef3b9fed337cc4d26ee684defd304fa0eac0`。引き続き有効。
- 両版とも `src/trottertracks/pauli_signal_entry.py` と `scripts/run_pauli_signal_entry.py` が、出力commitでその固定版とbyte一致することを検査する。新旧の混合は拒否。
- Aの説明：`08812cd30cf46c73346695cc20b6552c8359314a`、`docs/research/pr_pauli_m5_k2_composition_20261116/REPORT.md`。

`handoff_source`は実装commitではなく、Aが組み立てて公開したMEAN_HANDOFFの40桁commit/pathを渡す。

```python
accepted = intake.accept_saved(root, integration=api, handoff_source=packet_locator)
actual_assembler = accepted['assembly_implementation_source']
for item in accepted['results']:
    assert item['signal_packet']['assembly_implementation_source'] == actual_assembler
```

新旧の組立てSOURCE、出力packetのsource、per-round numerical SOURCE、原carrierを区別する。旧packetを新SOURCEへ再ラベルしない。従来のLD1/LD2・LD3・LD5固定packetも保全し受け入れる。

## 厳密ΔPF=1/5の扱い（重要な差分）

| 段階 | 状態 | SOURCE・理由 |
|---|---|---|
| A signal providerの対応 | m2～m12はUNSUPPORTED | 新A `round_support`：ACTUAL_STEP_OR_T_NOT_BINARY64_EXACT |
| Aの新信号 | 未取得。該当roundは計算に入れない | 今回は実signal 0件。旧complete packetの受入は、新providerの対応資格ではない |
| Bの保存packet受入 | SOURCE/physical/law/辞書/元carrierが一致する保存packetをmetadata受入 | 新旧とも資格はUNQUALIFIEDのまま。人工complete packetも実signalとして登録しない |
| Cの価格join | 独立したUNKNOWN/拒否 | C `c628e9e8447c11c984064b922cbce6c52dc72d48`のexact-step physical bridge。AのUNSUPPORTEDと別fieldで保持 |

B2はA PLANのround別`provider_support`・`provider_reasons`／`unsupported_rounds`を先に保持する。Cの`C_plan_price_bridge`や価格needsへ理由を上書きして混同しない。A未対応のroundをC側の価格取得で解消できるとは扱わない。binary64の0.2を厳密1/5の代替として自動採択しない。

## 正規化子と資格

新Aの`normalizer_budget`はsigned piece×要求roundに必要なscopeを数える関数。Bは実normalizationを再計算しない。Aのm5_best/K2人工合成検証は引用できるが、H4の実signal取得や精度資格に昇格しない。K4/K6の範囲へ拡張しない。

関連21 test PASS（[TESTS](TESTS.txt)）。新旧保存4構成、変更κ/ΔPF/LD4の通常intake、Aの新reasonとCの別拒否、元packet/資格保全を確認済み。今回のPauli接続では実signal/compile/Q3 0件。SOURCE付き[機械可読locator](INTAKE_LOCATOR.json)を参照し、新たなPauli科学実行へは進まない。
