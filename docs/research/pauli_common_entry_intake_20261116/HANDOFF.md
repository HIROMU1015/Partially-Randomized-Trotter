# B2・A向け：共通PR-Pauli packetの通常intake

## 固定locator

| 項目 | SOURCE / path |
|---|---|
| Bの実装 | `bd42dd1e95eb1d4152216d62792ad3421b803f7c` |
| B2が呼ぶ入口 | `src/trottertracks/rpe_pauli_signal_intake.py` : `accept_saved(root, *, integration, handoff_source=None)` |
| C接続adapter | 同じB SOURCEの `src/trottertracks/rpe_c_interfaces.py` : `interfaces(root).pipeline` |
| Aの固定組立て実装 | `93a5ef3b9fed337cc4d26ee684defd304fa0eac0` : `src/trottertracks/pauli_signal_entry.py`、`scripts/run_pauli_signal_entry.py` |
| Aの辞書helper | `8c8ce542e9fc25ead709fe6b0eb71b87ba7434ca` : `src/trottertracks/df_head_pauli_tail_ld3_signals.py` : `b_intake_dictionary_identity` |
| Aの説明 | `715af778833b8aaf84d26d467cf3f927771a84c9` : `docs/research/pauli_signal_entry_20261116/HANDOFF.md` |
| C planner/guard本体 | 従来どおり `c628e9e8447c11c984064b922cbce6c52dc72d48` |

機械可読の[INTAKE_LOCATOR.json](INTAKE_LOCATOR.json)も参照。B2は自分のroute view/checkoutでこのB実装を使う。他担当worktreeへ書かない。

## 呼出し

```python
from pathlib import Path
from trottertracks import rpe_pauli_signal_intake as intake
from trottertracks import rpe_c_interfaces

root = Path('/absolute/path/to/your/repository')
api = rpe_c_interfaces.interfaces(str(root)).pipeline
handoff = dict(commit='<Aが公開したpacketの40桁commit>',
               path='<repository relative directory>/MEAN_HANDOFF.json')
accepted = intake.accept_saved(root, integration=api, handoff_source=handoff)
item = accepted['results'][0]
configuration = item['configuration']
signal_packet = item['signal_packet']
signal_input = item['signal_input']
```

handoff_sourceは**出力JSONのGit locator**であり、Aの実装SOURCEへ置換しない。出力commit内でAの上記2fileが固定SOURCEとbyte一致している必要がある。Aの`--assemble`後にpacketをcommitし、そのcommit:pathを渡す。prefix/κ/ΔPFごとにBの定数や許可リストを書き足さない。Aの実装自体が変わる場合は、新SOURCEへの適合確認が別に必要。

configuration_sourceがnullのAのlocal-JSON/同一保存再利用modeも許容し、configurationは出力packetのSOURCEとcanonical検査に束縛される。別locatorがある場合は必ず照合する。configuration_sourceのselector（例：`['packets', 0, 'configuration']`）を解決し、packetのconfigurationと一致することを検査する。descriptorの辞書locator、辞書hash/bit順/λ/scalar、H/state/snapshot、全roundのcandidate ID・law・q/R/event/allocationは既存bind_packetが照合する。各行のmean_sourceから原carrierを読み、embedded mean_recordと完全一致させる。normalization/scalarは再適用しない。assembly_implementation_sourceはAの実装SOURCE、sourceは出力packet、implementation_sourcesは各roundの元数値SOURCEである。

既存LD1/LD2・LD3・LD5の固定HANDOFFも引き続き同じAPIで受け付ける。保存4構成を共通入口のpacket関数で再組立てた通常intakeと、従来intakeのsignal viewが一致することを確認した。

## signal-onlyと価格bridgeの区別

`integration=api`はC.plan/provider_binding_checkも実施する。人工κ×3/2、binary64 ΔPF=0.2、未登録LD4の辞書locatorでここまでPASSした。

厳密ΔPF=1/5の例では、signalのSOURCE/physical bindingは通るが、Cがcanonical有理数の実stepとphysical candidateのbinary64実stepをexact比較する別guardにより`Pauli physical recipe differs from B canonical round`となる。今回この比較規則は変えていない。signalだけを受け入れる既存APIの使い方は次のとおり。

```python
from types import SimpleNamespace
accepted = intake.accept_saved(root,
    integration=SimpleNamespace(signal_input=api.signal_input), handoff_source=handoff)
# results[0]['C_plan_price_bridge'] == 'UNKNOWN_PAULI_PHYSICAL_ID_ADAPTER_REQUIRED'
```

この場合C価格joinは未確認のまま。正式Gのguardを迂回したことや価格適格を意味しない。C側の実step表現bridgeが必要な具体的入力として引き渡す。B2はUNKNOWNを保持し、signal受入済みと価格join済みを混同しない。

## 受入結果とSTOP

`configuration`、`signal_packet`、`signal_input`は既存の形式。readiness/statistics_executedは診断を起動しない。meansはUNQUALIFIED、sufficient_N/global_RMSE_bound/G/winnerはnullである。実signal・compile・Q3は0件。decoder/確率law/target/lossの採択、N探索、価格算術をintakeから開始しない。

追加5＋回帰26＝31 test PASS。[REPORT](REPORT.md)と[TESTS](TESTS.txt)参照。新fixtureはtemporary Git内のみで削除し、実signal catalogには入れていない。ロードマップへの差分は「A共通packetの組立てSOURCE障壁を解消、exact-step C価格bridgeは別不足」。本体更新は統合担当に委ねる。mandatory STOP。
