# D → C／B：段階分離した決定論PF価格callerの受け渡し

branch `codex/d-price-records-c-bridge-20261114`、固定SOURCE `1fd47847b3300395f1b0c0de1868ed7b83905497`。

- 関連文書：[REPORT](REPORT.md)／[実行手順と条件](NEXT_EXECUTION.md)
- CとBのbranch・worktreeには触れていない。
- 価格は未取得である（Stage 1は未開始）。

## C担当へ

### CALIBRATION packet（Stage 1の後に公開予定）

置き場所は、各routeの `…/calibration/run01/` である。

| file | 内容 |
|---|---|
| `PRICES.json` | 新規の価格record。route別に、Full PFは10件、Truncatedは12件。分類は `DIRECT_COMPILED_DETERMINISTIC`／`DETERMINISTIC`、`record_schema=D_PRICE_RECORD_B_KINDS_V2`。`source.stage='CALIBRATION'` |
| `SIDECARS.json` | 公開commitを埋めたsidecar。`SIDECARS_UNBOUND.json` はplaceholder付きの工程記録なので、使わないこと |
| `IMPLEMENTATION_EVIDENCE.json` | Cのguardが読む固定assertion |
| `GUARD_CHECK.json` | Dが実行したC `c628e9e8` guardの結果。技術互換性probe policyによるもので、正式な比較policyではない |
| `C_INPUTS.json` | model group単位の入力（下記） |
| `RESULT.json` | cellごとのtrial（compiled全metric、作用gate、wall秒、peak RSS、wrapperのfingerprint／qubit数）、worker数とresource、SOURCE |

### `C_INPUTS.json` の形式

model group単位で、keyは `route|formula|width_over_h|axis`、route別に6 groupある。各groupの値は次のとおり。

```json
{"model_group": [route, formula, width, axis], "stage": "CALIBRATION", "features": ["one", "q"],
 "new_records": [PRICE_CALIBRATION_RECORD_V1, ...],
 "saved_records": [{"record": PRICE_CALIBRATION_RECORD_V1, "origin": "C_ADMITTED_SAVED_RECEIPT_NOT_RECOMPILED", "references": {...}}]}
```

- `new_records` は、Dの `fit_inputs` → Cの `price_model_validation.record` の出力である。
- `saved_records` に入るのは、Full PF M5の2 group（X、Y）だけである。中身はCのbundle `3368805575` の `SAVED_CALIBRATION_INPUTS.json` のrecordをそのまま使い、所在（bundleのPRICES、`2fa88065` のSIDECARS／GUARD_CHECK）を `references` に付けた。原cohort `EXPLORATION` と原SOURCE（`155529e20`）は変えていない。
- fitの入力は `[s['record'] for s in saved_records] + new_records` で、各groupはm4とm5の2点になる。
- Dは公開前に、各groupが同一route・formula・幅・軸・stageであること、およびCの `_domain` が一致することを確認する。
- `cell_id` には軸が入らない（X/Yで同じ文字列）。軸はrecordの `axis`（および `domain.axis`）で区別する。

### HOLDOUTのgateが前提とする、Cの凍結形式

`--frozen-models COMMIT:PATH` で、Cの固定commit上のJSONを指定する。

- 文書の構造は問わない。ただし、Cの `price_model_validation.fit` の出力dict（`model='PRICE_LINEAR_FEATURE_MODEL_V1'`）を、そのまま含むこと。
- route別の6 groupのそれぞれについて、次をすべて満たすmodelがちょうど1つあることを要求する。
  - `status='CALIBRATED_UNVALIDATED'`、`features=['one','q']`、`qualified_for_candidate=false`、係数は `one` と `q`。
  - `design_source` がある。
  - `domain` が公開済みの入力と一致する。
  - `calibration` の (cell_id, source) が、公開済みCALIBRATIONの `C_INPUTS.json` のgroupと完全に一致する（m6を含まない）。
  - 1つのmodelの中で、公式や幅が混ざっていない。
- `validate` を適用した後のmodel（status変更済み）は受理しない。holdoutの判定結果は、別のファイルに保存すること。
- 許容差・分母・policyのSOURCEは、Cが同じcommitに記録する想定である。Dはその値を検査しない（採択はユーザーの判断）。

### HOLDOUT packet（Stage 2の後）

- 構成はCALIBRATIONと同じで、置き場所は `…/holdout/run01/` である。`source.stage='HOLDOUT'`。
- `C_INPUTS.json` には、各groupのm6を1点だけ入れる（`role='HOLDOUT'`）。
- `RESULT.json` の `holdout_gate` に、次を記録する。
  - 使用した凍結modelの所在、SHA-256、係数。
  - CALIBRATIONの7ファイルの実行前のSHA-256。
- `calibration_after` には、実行後に再照合したCALIBRATIONのSHA-256を記録する。
- 判定（`validate`）はCの担当である。DはC_obsとC_predを比較しない。

### C guard

- Dのbindは、Cの受入時と同じ形のprobe policyを使う。
  - id：`TECHNICAL_COMPATIBILITY_PROBE_ONLY_NOT_ADOPTED`
  - policy SOURCE：C `c628e9e8` のguard
  - 方式別の設定：native `H46_LONG_TIME_NATIVE_COMPOSE`、`NOT_APPLICABLE_NO_RTE`。scalar quantityは、Full PFが `HAMILTONIAN_CONSTANT`、Truncatedが `HAMILTONIAN_CONSTANT_PLUS_B0I_TAIL_IDENTITY`。
- 正式な比較policyで再検査するのは、Cの担当である。

## B担当へ

- 今回は、`paper_nominal_cost`、Q3、`G_qualified` には接続していない。
- 価格recordは、Bの `PRICE_KINDS` に従っている。
- 新しいdecoderのcanonical IDへは付け替えていない。各方式の元のphysical IDと価格SOURCEを維持した。

## 保全したもの

- **D固定SOURCE**：`62ae1bd5`（全セル一括のcaller）は、このbranchの履歴に残る。`04bead11`／`8b28dce0` は旧branchに残る。
- **Cの成果**：Cのbundle `3368805575`／`2fa88065`、およびguard／model検証 `c628e9e8` は、読むだけで変更していない。
