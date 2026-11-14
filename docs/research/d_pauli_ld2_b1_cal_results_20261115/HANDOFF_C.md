# C向けHANDOFF：LD2 B1-CAL有限cohort価格10 record

LD2 CALIBRATION5 cell、8 fresh normal path＋S66からのpaired1 pathを取得した。予定18・再利用0・実compile18、全18 PASS/FAIL0/取得UNKNOWN0。D2の公開receipt readerは10 record・18個別receiptでPASS。実装SOURCE `13ab1acfaebe5f33958e1db67862f5c45725ab91`、cell/physical ID/cohort/seedの正本 `043c7567451e8f502a1cf0f425d32efe0a50f5dc`。本HANDOFFの公開commitは実行codeと別である。[全結果・資源・gate](REPORT.md)、[RAW_PUBLICATIONS](RAW_PUBLICATIONS.json)に固定SHAを示す。

## 1. 受領物と有限cohort値

| cell | 軸 | n_paths | 有限cohort RZ値 | 個別receipt |
|---|---|---|---|---|
| S66 | X | 2 | 2642 | `p0X`, `p1X` |
| S66 | Y | 2 | 2642 | `p0Y`, `p1Y` |
| S526 | X | 2 | 11045 / 2 | `p0X`, `p1X` |
| S526 | Y | 2 | 11045 / 2 | `p0Y`, `p1Y` |
| S4205 | X | 2 | 28854 | `p0X`, `p1X` |
| S4205 | Y | 2 | 28854 | `p0Y`, `p1Y` |
| Q8R66 | X | 2 | 37935 / 2 | `p0X`, `p1X` |
| Q8R66 | Y | 2 | 37935 / 2 | `p0Y`, `p1Y` |
| P2R66 | X | 1 | 4977 | `p0X` |
| P2R66 | Y | 1 | 4977 | `p0Y` |

機械可読の入口は [C_HANDOFF.json](C_HANDOFF.json)。`records`は元の公開後binding10 record、`sidecars`は対応する10 sidecar、`cells`はn_paths・q/R/T・m=None・元physical ID・cohort/seed・元RESULT locator・各cellの保存reader checkの索引、`individual_compile_receipts`は18個別receipt/元path/fingerprint locatorである。各cellのraw `C_INPUTS.json` 等を上書きせず、`C_INPUTS_PUBLISHED`/`SIDECARS_PUBLISHED`を別保存した。

kind=`DIRECT_COMPILED_FINITE_SAMPLE`、quantity=`FULL_WRAPPER_MEAN_COMPILED_PRICE`、expectation=`FINITE_COHORT_ONLY`。値は測定付きfull controlled Hadamard wrapperのcompiled RZ数の有限cohort平均。X/Yは同一pathの別軸。P2はS66親2本と相関する1 derived pathで、独立の追加pathではない。B0/Wを標本に含めない。Cが独立n=9や18と扱うことを避け、親子対応を維持する。

## 2. Cの読取入口（fitを呼ばない）

D2 branchのrepository rootから、既存13abのAPIだけで照合できる。保存された各 `C_CELL_CHECK.json` はD2が今回実行済の結果である。下記は同入口をCが使用する例で、C別worktreeの変更は必要ない。

```python
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'scripts'))
import run_pauli_b1_prices as B1

base = Path('docs/research/d_pauli_ld2_b1_cal_results_20261115')
delivery = json.loads((base / 'C_HANDOFF.json').read_text())
plan = json.loads((base / 'JOBS/S66.json').read_text())['plan']
for item in delivery['cells']:
    cell = next(c for c in plan['cells'] if c['cell_id'] == item['cell_id'])
    records = json.loads(Path(item['records_path']).read_text())
    sidecars = json.loads(Path(item['sidecars_path']).read_text())
    check = B1.read_calibration_records(plan, cell, records, sidecars)
    assert check['status'] == 'CELL_RECEIPTS_CHECKED_NOT_SCIENTIFIC_QUALIFICATION'
# fit / freeze / holdout は呼ばない。
```

各cellの `bind_calibration_records(publication_commit=RAW_PUBLICATIONS[cell]['commit'], result_path=RAW_PUBLICATIONS[cell]['path'])` は実行済。そのraw公開SHAを読んで `source.publication` と `source.implementation_evidence` を固定し、`source.commit`（実行code SHA）を維持した。readerは元physical request、辞書、cell/axis/path/cohort/seeds、作用gate、実装SOURCE、metrics/fingerprintと元compile receiptの対応を照合する。statusはmetadata/receipt整合の意味で、law_expectation/model_qualifiedはfalse。QPYの再compile/再simulation・C fitはこの入口では行わない。

## 3. pairedの固定親locator

P2R66は公開済S66 RESULTのpath0→path1を直接連結した。親実行SOURCE `355e7074df806b60a31160a51899ec15d6b54e33` と親raw公開SHA `ff536ed2a6effc6c01899a7dcde76e5361448b85` を区別する。

```json
[
  {"commit": "ff536ed2a6effc6c01899a7dcde76e5361448b85", "execution_commit": "355e7074df806b60a31160a51899ec15d6b54e33", "path": "docs/research/d_pauli_ld2_b1_cal_results_20261115/run01/S66/RESULT.json", "selector": ["child", "paths", 0]},
  {"commit": "ff536ed2a6effc6c01899a7dcde76e5361448b85", "execution_commit": "355e7074df806b60a31160a51899ec15d6b54e33", "path": "docs/research/d_pauli_ld2_b1_cal_results_20261115/run01/S66/RESULT.json", "selector": ["child", "paths", 1]}
]
```

固定callerがSOURCE/request/辞書/cohort/index/seed/順序/gate/phaseをbuild前に検査し、sampler0回、保存event直接連結でX/Yを個別compileした。総T=.2のscalar一回、`PAIRED_STEP_METADATA_IDENTICAL`。子rawは `2669ad11da9d116f2834ee49a9f4a5b0654d91ef`、子実行code HEADは `74955a11ce955f49341183bd5aa933f6af707104`（実装は13abとbyte一致）。[PAIR job](JOBS/P2R66.json)と元RESULTに同じ参照を保存している。

## 4. 受入済みの範囲と次の依存

D2実作用/compiled-vector gateは9 path全PASS、公開readerは10 record全PASS。これはC自身のfit・係数凍結・モデル科学受入の完了を意味しない。今回fresh10650 eventとpaired replay132 eventはすべてorder0、D2 aggregate phase0。実LD2 order2 product・非自明位相は未観測UNKNOWN。過去26人工/toy PASSを再利用しただけで、toy位相例を実観測にしない。

今回の終了点は有限cohort価格の引渡し。次は**別途指示によるC fit・凍結 → さらに別指示で独立m5/m6 holdout6 compile**。Pauli holdout許容差は未採択、決定論PF2%は移送しない。全lawの期待価格・CIを支える証拠は今回の小cohortからは得られていない。LD1、B1-V、E[C]・N・G、5方式総費用順位は未判定。

ロードマップへの影響は「LD2 CAL入力10 recordの引渡し完了、C fit前のreceipt整合確認済み」まで。ロードマップ本体は統合担当が更新する。本作業では編集しない。旧raw/計画/STOP/SCOPE_DEVIATION/log保全、最終pushとremote一致確認後に**mandatory STOP**。
