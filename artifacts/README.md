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
- [`finite_rte_phase_amplitude/2026-09-26/`](finite_rte_phase_amplitude/2026-09-26/)：
  FR-1のfingerprint付き2×2 fixed-grid結果と実行時FR-0/FR-1契約の凍結copy。
  G2不通過の`GO_FR2_MECHANISM_ONLY`であり、FR-2やH4結果ではない
- [`pr2_pr3_minimal_pilot/2026-09-27/`](pr2_pr3_minimal_pilot/2026-09-27/)：
  事前登録済みPR-2 H4 rank圧縮残差とPR-3固定2-qubit外挿のfingerprint付きlocal結果。
  PR-2主題候補選択後の強制STOPを含み、最終総costまたは外部CI証拠ではない
- [`pr2_s1_s3_preregistration/2026-09-27/`](pr2_s1_s3_preregistration/2026-09-27/)：
  PR-2 S1--S3の実装監査、固定条件、入力・prefix identity blocker、予定task上限を記録したdry-run manifest。
  数値結果ではなく、実行許可はfalse、全実行countは0である
- [`pr2_s0_s1_validation/2026-09-28/`](pr2_s0_s1_validation/2026-09-28/)：
  source commit `c644925`のtest log、development/held-out input snapshot、development byte-hash不一致で
  `STOP_INPUT_REPRODUCTION_MISMATCH`となったS0 artifact。S1結果やheld-out signal/cost/rankingは含まない
- [`pr2_s1_s3_preregistration/2026-09-28/`](pr2_s1_s3_preregistration/2026-09-28/)：
  外部レビュー後amendment v2のmachine-readable dry-run manifest。補正後estimand、S1軽量化、stage gateを
  記録し、S0実装だけを許可する。S0/S1実行countは0である
- [`pr2_v4_s2_development/2026-09-29/`](pr2_v4_s2_development/2026-09-29/)：
  別系列のV4通過後に実行したH4 1.00 Å development-only S2結果。引用対象は結果JSONだけで、
  SQLite cacheとlogは再開・監査用でGit管理しない。held-out NPZ loadは0、S3は未承認である
- [`pr2_matched_accuracy_m1_b1_execution/2026-09-30/`](pr2_matched_accuracy_m1_b1_execution/2026-09-30/)：
  M1-Aで固定した194 random＋16 baseline cellの12,448-wrapper compile mapと完了marker。`.runtime`の
  checkpoint/cacheは監査用で、科学的な引用は軽量result JSONと検証報告を使う
- [`pr2_matched_accuracy_m1_b1_result_validation/2026-10-03/`](pr2_matched_accuracy_m1_b1_result_validation/2026-10-03/)：
  全checkpoint/cache再集計、actual frontier、旧selector、fixed-q=8、proxy、状態準備感度と
  `CONTINUE_RESOURCE_STUDY`判断を含むlocal validation artifact。held-out accessは0
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
