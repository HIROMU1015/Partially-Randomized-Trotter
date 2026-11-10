# A0受入API・公開後の最小取込み

[REPORT](REPORT.md)／[固定24件とC1 lane](INTAKE.json)。本成果は診断の受入準備だけであり、実A0結果は未消費。

## API

```python
from trottertracks.rpe_normalization_intake import A0Intake

intake = A0Intake(repository_root)  # 固定Git metadataのみ
ready = intake.readiness()        # normalization/N/C/Gを呼ばない

# Aの成果公開後、指定されたimmutable commit/pathを使う。
accepted = intake.read_published(
    {'commit': published_full_sha, 'path': handoff_path},
    {'commit': published_full_sha, 'path': result_path},
)

# C1 consumerが明示する同一SOURCE・元A requestで接続する。
cid = '4269eafc8c0ef8922c23'
link = intake.for_c1(
    accepted, cid,
    request=intake.expected[cid]['request'],
    normalization_source=accepted['scientific_source'],
)
```

in-memoryの人工/mock用には`accept(handoff, result=None)`がある。`result=None`でも元HANDOFF/24件の欠測を保持するが、保全証拠不足として共有資格はfalse。`for_c1`は同一条件確認後も、`share_eligible`がfalseなら未資格診断として返す。呼出側がこのflagと元statusを保持する。実際のC producerのrequestはそのconsumerが渡し、例のように期待requestを生成できることだけを実計算の条件一致の証拠にしない。

| 出力field | 意味 |
|---|---|
| `completion` | COMPLETE_RECORDED / PARTIAL / STOP / COMPLETE_UNQUALIFIED |
| `rows[].expected` | 固定PLANの元candidate/identity/law。未着でも全24件を維持 |
| `rows[].record` | 元診断recordの無計算copy。未着はnull |
| `rows[].normalization_status` | Aの計算/精度status。nullとNOT_EVALUATEDを保つ |
| `normalization_evidence_usable`, `evidence_blocks` | root/24件の数値記録/保全/元資格flagの診断消費条件 |
| `diagnostic_share_eligible` | 診断共有だけの資格。科学的PASSやN/C/Gの資格ではない |
| `original_handoff`, `original_result` | 元status/errors/null/精度表示/保全証拠のcopy |
| `N_records`, `C_records`, `N_C_G` | 常に空/空/null。費用pipelineへのN/C入力として使わない |

入力はA準備commit`6a75ebf96`のschemaに固定。resultはHANDOFFと同じSOURCE/status/recordsを持ち、`source_files`/`input_hashes`が固定PLANと一致すること。SOURCE/input保全がfalseならそのSTOPを保持し、flagを上書きしない。Git科学SOURCE9件とtail SOURCEは照合するが、live producer/source/inputは開かない。candidate/law/SOURCE不一致などで拒否された入力は、元公開artifactを保全して差分を確認する。

Aのlaw宣言をBの`IDEAL_FINITE_RTE_MEAN`へ置換しない。readinessは同一identityに対応する`B_provider_law_binding`を参照として併記するだけ。将来のN/C provider recordは別の科学証拠と元requestへのbindingが必要。

## CLI・テスト

受入準備のみ：

```bash
/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B \
  scripts/prepare_rpe_normalization_intake.py --output-dir <fresh-B-dir>
```

公開後の受入は同じCLIに`--handoff-commit <full-SHA> --handoff-path <repo-relative-HANDOFF.json> --result-path <repo-relative-result.json>`を指定する。二つのpathは同じ公開commitを読む。入力は既存`SavedArtifacts`で読むだけで、実行・retry・次段階起動optionはない。Aの実行中directoryや進捗fileは指定しない。

```bash
/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B \
  tests/check_rpe_normalization_intake.py
```

[TESTS](TESTS.txt)は新8＋既存39件のPASSを記録。INTAKE.jsonは保存PLAN由来のmetadataで、実normalization値や人工診断値を含まない。

## 次の担当と境界

1. **Codex A**：承認済みA0の結果を公開commitに固定し、HANDOFF/resultのpathと科学SOURCEを示す。BはAの実行中領域に触れない。
2. **Codex B**：公開SHA/pathを受けた後にpassive intakeのみ実施し、保全/欠測/STOP/数値未資格を残す。新計算・再解析は追加しない。
3. **Codex C**：C1の元8セルを維持し、同じSOURCE/requestの重複2候補だけ診断共有できる。H4 R8/q8・q16はCの明示laneのまま。A0の24件へ増設しない。
4. **ユーザー**：signal/N、新R/長q価格、全law/共分散、全RPE成功、実G/R最適化に進む場合は対象と有限scopeを別途判断する。A0完了をその許可に読み替えない。

現在の担当はcommit/push・remote照合後mandatory STOP。科学結果の到着待ちや追加科学実行を自動起動しない。
