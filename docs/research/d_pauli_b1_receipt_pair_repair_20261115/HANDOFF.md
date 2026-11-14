# C/D2引渡し：非configuration価格の公開証拠とpaired直接連結

実装SOURCE **`13ab1acfaebe5f33958e1db67862f5c45725ab91`**、branch `codex/d-pauli-b1-caller-prep-20261114`。[REPORT](REPORT.md)／4 suiteの保存logを同dirに収録。人工/toy **26 PASS**、実H4科学call/compileは0。実価格のC受入・LD2の実order2観測・fit/holdout/E[C]/Gは未判定。技術公開後mandatory STOP。

## 1. 既存構成との対応

対象は元LD2 B1-CALのS66、S526、S4205、Q8R66、P2R66。q/R/T・辞書・cohort/seed・m=None・元physical IDは計画`043c75674`のまま。[元cell/seed表](../d_pauli_b1_caller_preparation_20261114/CAL_AFTER_W_METADATA.json)を参照する。同表の旧実装説明「REGENERATE_WITH_PARENT_SEEDS_THEN_COMPARE」は履歴であり、今回のSOURCEでは本HANDOFFの直接連結に置き換わる。PROPOSALS/JOB_TEMPLATES・旧raw/STOP・SCOPE_DEVIATIONとログは改変していない。

旧W `1e2639a1b`（raw `6c4709f85`）は配線の前提だけで、CAL標本には使わない。元B0/W normal sampler・DF head、Pauli D2 lowering、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、測定付きfull wrapperと総Tでscalar一回を保つ。B0 defaultの結果同値はtoyで確認した。Wの科学的な再実行・再認定はしていない。

## 2. 実取得後に出力される構成

| file / field | 接続する内容 |
|---|---|
| `RESULT.json → child.paths[i]` | 完全なrequest、dictionary_identity、cohort/index/seeds、作用・compiled-vector gate、bit順、event/D2 metadata、X/Y metrics/fingerprint |
| 同pathの`result.compile_receipts[cosine/sine]` | native.compile_cachedが返した個別compile record。cache hit時も保存compiled QPYを読んで当該stateのcompiled-vector gateを通す経路 |
| `COMPILE_RECEIPTS.json → p0X,p0Y,...` | 既存private cache JSONから転記した公開receipt。cell/axis/index/request/辞書/cohort/seed/実行SOURCEと、compiler/metrics/fingerprint、private QPY原本の所在・hash |
| `C_INPUTS.json` | X/Yの`PRICE_CALIBRATION_RECORD_V1`。有限cohortのrz_count、役割、特徴量、law/profile/domain、n_paths、実行code SHA。元rawとして保全 |
| `IMPLEMENTATION_EVIDENCE.json → X/Y` | 当該record、物理request・辞書、native/wrapper/D2/decoder等SOURCE、scalar・測定、各pathと公開compile receiptのlocator |
| `PRICES.json` | 各path/axisのdirect finite price。canonical round recordへは偽装しない |
| `SIDECARS_UNBOUND.json` | 公開前のlocator（commit=null）。受入完了として使わず原本を保持 |

QPY原本は既存の専用private cacheを保持し、Cの読取入口は公開JSON receiptを照合する。C読取はQPYの再simulationや再compileを行わない。科学的な回路資格は元pathのgate/SOURCEに結び、単なるmetadata整合を全law資格にしない。

## 3. 公開後bindingとCのfit前読取

入口は`run_pauli_b1_prices.bind_calibration_records`と`read_calibration_records`。処理本体は`src/trottertracks/pauli_b1_receipts.py`で、project物理moduleをimportせずJSON/Gitと供給されたcountだけを扱う。下記は**将来の実取得・raw公開後の入力例**であり、現時点に実CAL結果は存在しない。

```python
# D2 SOURCEをcheckoutしたrepository rootで使用する。
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd() / 'scripts'))
import run_pauli_b1_prices as B1

plan = B1.proposal(2, 'B1-CAL')  # metadata only; adopted=Falseのままで読める
cell = next(c for c in plan['cells'] if c['cell_id'] == 'S66')

# 以下2値は実取得後、公開された不変raw locatorで指定する。
RAW_RESULT_COMMIT = 'REPLACE_WITH_40_HEX_RAW_PUBLICATION_SHA'
RESULT_PATH = 'docs/research/YOUR_NEW_RUN/S66/RESULT.json'

bound = B1.bind_calibration_records(
    publication_commit=RAW_RESULT_COMMIT, result_path=RESULT_PATH)
checked = B1.read_calibration_records(
    plan, cell, bound['records'], bound['sidecars'])

# rawを上書きせず、別ファイルとして公開する例。
B1.write(Path(RESULT_PATH).parent / 'C_INPUTS_PUBLISHED.json', bound['records'])
B1.write(Path(RESULT_PATH).parent / 'SIDECARS_PUBLISHED.json', bound['sidecars'])
B1.write(Path(RESULT_PATH).parent / 'C_CELL_CHECK.json', checked)
```

全5 cellを個別に読取る。一軸5 recordが揃った後に使用できる既存fit入口は`finite_price_design.fit_identified(..., axis='X'/'Y')`→C固定`price_model_validation.fit`である。**今回fitを呼んでいない**。fitは別の科学実行許可の範囲で行う。

実行codeと結果公開のlocatorは別々である。

```text
record.source.commit                       = 実行code SHA
record.source.publication                  = {raw公開commit, C_INPUTS.json, [軸位置]}
record.source.implementation_evidence      = {raw公開commit, IMPLEMENTATION_EVIDENCE.json, ["X" または "Y"]}
sidecar.result                            = {raw公開commit, RESULT.json, []}
checked.checked[軸].paths[i].path          = {raw公開commit, RESULT.json, ["child","paths",i]}
checked.checked[軸].paths[i].receipt       = {raw公開commit, COMPILE_RECEIPTS.json, ["p0X" 等]}
```

読取入口は元rawを開き、cell/axis/path集合、requestとm=None/physical ID、辞書、cohort/seed、役割・特徴量・domain・quantity・n_paths、作用gate、metrics/fingerprint/compile receipt・平均の対応、実装証拠、実行/公開SOURCEを照合する。軸取違え・receipt欠落・fingerprint不一致等を拒否する。返却statusは`CELL_RECEIPTS_CHECKED_NOT_SCIENTIFIC_QUALIFICATION`、law_expectation/model_qualifiedはfalse。C別worktreeの変更は不要。

## 4. paired P2R66の入力例

将来のpaired jobの`parents`はinline eventの再生成用ではなく、**保存S66結果の公開locatorをpath0、path1の順**に指定する。

```json
[
  {
    "commit": "REPLACE_WITH_S66_RAW_PUBLICATION_SHA",
    "execution_commit": "REPLACE_WITH_S66_EXECUTION_CODE_SHA",
    "path": "docs/research/YOUR_NEW_RUN/S66/RESULT.json",
    "selector": ["child", "paths", 0]
  },
  {
    "commit": "REPLACE_WITH_S66_RAW_PUBLICATION_SHA",
    "execution_commit": "REPLACE_WITH_S66_EXECUTION_CODE_SHA",
    "path": "docs/research/YOUR_NEW_RUN/S66/RESULT.json",
    "selector": ["child", "paths", 1]
  }
]
```

これは入力形式の例で、承認済みjobではない。`execution_commit`は親code、`commit`は親結果の公開SHA。doc/raw公開でHEADが前進しても、caller/core/connection/reader/decoderの使用SOURCEが親のcodeとbyte一致することを検査する。親の元SOURCEを子のSHAへ付け替えない。

`execute`は親SOURCE・request・辞書・cohort/index/seed・順序・gateを読取確認してから子へ渡す。`acquire_path`は保存eventを既存`h46_same_event.decode_event`で復元し、係数/label/order/product/normalization・D2 token/phase owner/aggregate phaseを検査してから既存B0 coreへ保存stepを渡す。childではこのevent復元・phase検査をcontext復元/preparationの前に済ませ、検査済stepをそのままacquire_pathへ渡す。pairedではsamplerを呼ばない。親full wrapperや親のscalar phaseを連結するのではなく、親event列をstep順に使い、scalarは子total Tで一回だけ適用する。X/Yはそれぞれの測定wrapperを生成する。

保存stepの`outer`は子の0,1に対応付ける。event順序・label・product・phase・seed・D2 metadataは保持し、生成後も親との完全なJSON対応を確認する。pairedと親は相関する同じ標本で、独立な新2標本とはしない。

## 5. 未判定とSTOP

人工の5 cell×X/Yと4-mode toyで確認したものと、実LD2で初めて確認できるものを区別する。実CALの入力/作用/phase・実compile・公開receiptへのC受入、order2/非自明位相の実観測、fit・モデル凍結・独立holdout・全law E[C]/CI・正式Gは未判定。Wでのorder0 component34/40のみという観測は変わらない。Pauli用許容差は未採択で、PFの2%を移さない。

ロードマップへの影響は「2点の限定技術修正はSOURCE固定・26人工/toy PASS、実LD2較正18は保留」に限る。科学的価格coverageは増えていない。次に実取得を選ぶ場合も、対象SOURCE・5 cell/18 compile・出力・one-shot条件を明示した別の実行指示が必要。較正→C fit・凍結→別指示で独立holdoutの順序を保つ。ロードマップ本体は無変更。

**今回の技術公開でmandatory STOP。V/CAL実行、実価格のC受入、fit、holdout、E[C]/Gへ進まない。**
