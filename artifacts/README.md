# artifacts の扱い

`artifacts/` は計算入力のsnapshot、機械可読な結果、図表用データ、キャッシュ、再開用状態を置く。
ファイルが存在することだけを根拠に、現在有効な研究結果とは判断しない。

## 証拠の入口

- [`validation_manifest.json`](validation_manifest.json)：result set、状態、生成コード、テスト、成果物の対応
- [`../VALIDATION_STATUS.md`](../VALIDATION_STATUS.md)：再現可能性、失効理由、未解決事項
- [`../docs/research/研究概要・現状.md`](../docs/research/研究概要・現状.md)：現在採用している結果の要約

- [`research_direction_energy_tail_pareto/2026-09-25/`](research_direction_energy_tail_pareto/2026-09-25/)：
  P-Dの固定expected tasksとexact two-block Pareto結果。負時間sampled RTEと内部`H_D`誤差は
  後続の現実化gateと区別して使う
- [`research_direction_pd_realization/2026-09-25/`](research_direction_pd_realization/2026-09-25/)：
  P-D現実化のv1/v2 expected tasksとD1--D3完了結果。v1は技術的KeyError前の凍結履歴で、
  数値判断にはv2 resultを使う
manifestは次で検査できる。

```bash
python3 scripts/check_validation_manifest.py
```

## 内容の区分

- 検証結果：検証名に対応するディレクトリ内の小さなJSON、CSV、metadata
- 入力snapshot：DF Hamiltonianや基底状態など、同じ検証条件を再現するための固定入力
- キャッシュ：再計算を短縮する中間生成物。研究上の結論ではない
- checkpoint/log：長時間計算の再開・監査用。通常はGit管理しない
- 旧結果：manifestまたは`VALIDATION_STATUS.md`でstale、invalidated、prose-onlyとされたもの

新しい数値を引用するときは、対応する検証文書に書かれた条件と限界も同時に確認する。

現在の長round候補に対する局所角度診断、イベント列長holdout、中央RTEブロックのschedule別
compiled-cost proxyは`rpe_delta_compiled_cost_validation/2026-09-20/`に置く。同ディレクトリの
SQLiteとcheckpointは再開用であり、引用する結果は検証済みJSONと
[`../docs/rpe_delta_compiled_cost_validation.md`](../docs/rpe_delta_compiled_cost_validation.md)を使う。
