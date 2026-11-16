# B2・Aへ：LD4/6のdescriptor入口→通常intake→total→Q3入力

## 固定SOURCEとAPI

- **B実装：`33e324b43690caeaa353ecdb9e3d42f3139198b6`**（本branchの後続文書commitからも同じ実装）。
- 通常受入：`src/trottertracks/rpe_pauli_signal_intake.py::accept_saved`。
- total：既存`src/trottertracks/rpe_total_binding.py::create / bind_signal`。
- Q3入力準備だけ：既存`rpe_signal_statistics_caller.prepare`（`T.K.prepare`）。**T.evaluate/Q.evaluateは呼ばない**。
- Aのdescriptor入口：`6e4b6bd6a491815c2a3cb8f809cbfc0240b60fdb`、`scripts/run_pauli_signal_entry.py`。`ebab4209798b6e2253ea227739673ab91ea92559`もentry module/runner/descriptor helperの3fileがbyte同一。
- B2照合対象：`d7a976a629a6db68197b24f8f86734e109c8bf5d`。当時のintake bd42とff9は同じcode。ff9だけへの変更では不足。
- C planner/guard：従来`c628e9e8447c11c984064b922cbce6c52dc72d48`を維持。

原configuration・P0辞書・descriptorの固定locatorと全13roundのoriginal/total physical IDs・lawは[SAVED_INPUTS.json](SAVED_INPUTS.json)と[READINESS.json](READINESS.json)。LD4 original `8bf6eb88…`、LD6 original `6b6e0f17…`を、Pauli LD3/LD5へ置換しない。

## B2側で必要な差分

1. 自分の新route viewを上記B実装へ固定する。B2の作業中worktree、原run、旧route viewはそのまま保全。`intake_view`、routeの`view/view_commit`、`signal_intake_source`を新Bへ揃える。
2. signal用checkout/`signal_entry_source`はA6e4版（または上記同code版）を使う。旧93aへSOURCEだけを偽装しない。
3. A入口の**plan/execute/assemble全て**へ、Aの準備結果のdescriptor envelopeを`--tail-descriptor <DESCRIPTOR.json>`として渡す。B2 d7の`rpe_split_stage.signal_support`とdriverはまだこの引数を渡していないので、単にSOURCE設定を変えるだけでは不十分。descriptorはprepared itemから受け取る一般入力とし、prefix別のif/allowlistを増やさない。
4. [B2_COMMON_METADATA_VERSION.patch](B2_COMMON_METADATA_VERSION.patch)の1行を、B2側の`common_metadata`へ反映する。実`common_case`は変更不要。`accept_saved`後の組立てSOURCEを `A.ENTRY_SOURCE`（旧1版）ではなく、検証済み集合 `A.ENTRY_SOURCES`で検査する。未知SOURCEはB intakeが事前に拒否する。このpatchはd7の関数に対して人工通し検査済み、B2の実branchへは未適用。

## Aの計画の起動例（実signalを計算しない）

A6e4の自分のcheckoutから：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B   scripts/run_pauli_signal_entry.py --plan   --config <LD4またはLD6の保存CONFIGURATION.json>   --tail-descriptor <対応する保存DESCRIPTOR.json> --out <新しい計画出力先>
```

今回の両planは全13round REQUEST、unsupported/reuseは0。descriptor envelopeはB2 d7の `docs/research/rpe_split_preparation_20261116/LD4/PREPARATION_P1.0.DESCRIPTOR.json`、LD6の `PREPARATION.0.DESCRIPTOR.json`。中身のsourceはLD4 c35495565、LD6 f3f47f7f9の元descriptorを指す。既存source/descriptor値を上書きしない。

**今回executeはしていない。** 実signal取得を許可された次回だけ、同じconfig/descriptorをexecuteへ渡し、raw commit後のassembleにも渡す。packetを新commitに公開して、そのlocatorを以下へ渡す。

## 通常B intakeとtotal/Q3入力の組立て例

```python
from pathlib import Path
from copy import deepcopy
from trottertracks import rpe_pauli_signal_intake as A, rpe_total_binding as T

root = Path('/absolute/path/to/your/new/B/view')
api = T.C.interfaces(str(root)).pipeline
packet_ref = {'commit': '<公開済みMEAN_HANDOFFの40桁commit>',
              'path': '<repository relative>/MEAN_HANDOFF.json'}
got = A.accept_saved(root, integration=api, handoff_source=packet_ref)
item = got['results'][0]
store = A.p.SavedArtifacts(root)
handoff = store.read(packet_ref)
config_ref = handoff.get('configuration_source') or dict(packet_ref,
    selector=['packets', 0, 'configuration'])
binding = T.create(root, item['configuration'], original_source=config_ref)
bound = T.bind_signal(root, binding, item['signal_packet'])
```

`got`と`signal_packet.assembly_implementation_source`は照合したA実装版、`source`は出力packet、`implementation_sources`はround別の元数値SOURCE。実装codeが同じ後続commitは固定実装版6e4として検査し、出力/元数値locatorを変更しない。辞書hash/bit順/λ/scalar/H/state/law/candidate/原carrierの検査は従来どおり。scalar/normalizationは再適用しない。

B2 `common_case`に渡すregistry entryは、上のconfig_ref、packet_ref、binding['id']、binding['total_configuration']['id']を登録するだけ。元receiptのphysical IDをtotal IDへ付け替えない。

Q3入力の準備は、B2が選んだwhole-shot配分・評価target・契約を渡して既存workerと同じSOURCE補完を行う（今回科学的なshot列は選んでいない）：

```python
allocation = deepcopy(case['allocation'])
allocation['source'] = run_source
contract = deepcopy(inputs['statistical_contract'])
contract['source'] = run_source
contract['simulation_policy']['source'] = run_source
contract['summary_options']['unresolved_loss']['source'] = run_source
kernel_inputs = T.K.prepare(item['configuration'], item['signal_packet'], [allocation],
    evaluation_target=case['evaluation_target'], statistical_contract=contract,
    integration=api)
# prepareまでで止める。Q3数値kernelを呼ばない。
```

B2の実common_caseと、上記1行の版判定を修正したcommon_metadataを人工fixtureで通した。既存target/loss/ideal IID lawは変更せず、UNKNOWN/UNQUALIFIED、sufficient_N/global_RMSE/G=nullを保持する。入力組立て成功は閉包・mass保存・MSE・十分Nの認定ではない。

## データ境界と次の不足

| LD | 実保存資料で確認済み | 人工のみで確認 | 実signalが必要なround |
|---|---|---|---|
| 4 | configuration・descriptor・P0辞書・13round schedule・A plan・total ID生成 | 完備mean packetの通常intake/C metadata/total signal view/Q3入力 | m0..m12 |
| 6 | 同上 | 同上 | m0..m12 |

参照したB2信号branch `762286b05`にはLD4/6の保存packetが無く、今回のfixtureはtemporary Git内だけに置き、実信号catalogに登録していない。A planのSUPPORTEDは「計算済み」「H4でgate PASS」ではない。価格26軸や較正も今回未評価。実signal→Q3→価格・Gは別々の不足として保持する。

26技術test PASS（[TESTS](TESTS.txt)）、新規実signal/compile/実Q3=0。ロードマップ本体は統合担当へ。差分は「LD4/6をprefixで拒否せず、新A SOURCEを使う技術受入完了。現実の26round mean取得とB2側driver/route設定更新は残る」。本接続作業はmandatory STOP。
