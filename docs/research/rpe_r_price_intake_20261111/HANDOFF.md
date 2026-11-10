# C1価格結果の取込み手順とN/C接続条件

[REPORT](REPORT.md)。本ファイルは、C1の公開結果をBへ取り込む担当（Codex BまたはClaude Code）向けの手順書である。

- 取込みは、ユーザーがその作業を指示した後に行う。
- 取込みは公開済みJSONのcopyで、新しい科学計算ではない。
- 取込み後も、実N×C/GやR比較は別途の明示許可が必要。

## 1. 公開後の取込み手順

```bash
cd <B worktree>
git fetch origin codex/r-variable-compiled-price-design-20261111
/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B \
  scripts/plan_rpe_r_price_intake.py --output-dir <fresh-dir> \
  --publication-commit <C1結果の40桁commit> \
  --run-dir docs/research/r_price_phase1_results_20261111/run01
```

- 出力は次の3つ：`READINESS.json`、`INTAKE.json`（受入結果）、`COVERAGE.json`（C1 recordを含む静的coverage）。
- 2026-11-11 03:00時点のC公開headは`2e6a0cdf5…`（Phase 1完了STOP）。今回は取り込んでいない。
- Pythonから使う場合：

```python
from trottertracks import rpe_r_price_intake as ci
intake = ci.C1PriceIntake(root)                       # 固定PLAN c596e88d、Cのvalidate_plan
accepted = intake.read_published(commit, run_dir)     # copyと検査だけ
prices = packet['prices'] + ci.records_for(accepted, packet['menu'])
cov = ci.coverage(packet['menu'], packet['shots'], prices, cohort_selection={},
                  intake=intake, accepted=accepted, a0=a0_intake)
```

`packet = b.prepare_menu(root, system, R_by_round)`。C1の行を含めるには、menuにH4 {3,4: R4/8}、H6 {4,5: R16/32}が必要。他のRやroundを含んでもよく、C1 recordはexact identityの行にだけ付く。

## 2. 必要な公開SHA/pathと結果field

run directory内で使うファイル：
- `PLAN.json`：固定PLANと同一であることが必要。
- `FROZEN_MODELS.json`：`plan_digest`、`validation_prices_observed=false`、`validation_refit=false`、`acceptance_threshold=null`、models[H4/H6]の`calibration_cells`・`point_roles`・`shared_calibration_group`・`qualified_for_candidate=false`。
- `CALIBRATION_EVALUATION.json`：`evaluations`（6セル）、`prices`（24件）、`full_rpe_resources=null`。
- `VALIDATION.json`：`frozen_model_commit`、`validation_refit=false`、`evaluations`（2セル）、`prices`（8件）、`total_CI`・`all_law_expected_price`・`full_rpe_resources`はnull。
- `RARE_DIAGNOSTICS.json`：`pairs`（4件）、`cohort_pooling=false`。
- `STOP.json`：`status`。

価格recordのfield：
- identity、law_binding、law_context、axis、measurement_axis、observable、outcome_convention、units
- classification（DIRECT_COMPILED_FINITE_SAMPLE／MODEL_PREDICTION）、cohort、n_paths、expectation=FINITE_COHORT_ONLY、mean.rz_count
- source（execution_source=`1a87e533…`、result_path、validationではfrozen_model_commit）
- model（name・q・domain・binding・compiler・coverage・qualified_for_candidate）
- uncertainty

`frozen_model_commit`のFROZEN_MODELS.jsonは、publication commitの同じpathとbyte一致する必要がある。`PATH_RESIDUALS.csv`、`CALIBRATION.json`、`VALIDATION.json`の`data`（path別のC/S）はBでは使わない。

## 3. calibration・validation・rareの区別

- 校正：保存校正は4セル、新Rの校正は2セル。
- 独立検証：2セル。Git凍結後の価格で、fitには入っていない。
- model予測：校正点と検証点を`price_kind`で分ける。
- rare：`rare_diagnostics`だけに入り、price recordには入らない。
- stageの受入状態は次の4つ。
  - `CALIBRATION_RECEIVED_VALIDATION_PENDING`
  - `VALIDATION_RECEIVED_RARE_PENDING`
  - `PHASE1_COMPLETE_RECORDED`
  - `STOP`
- 欠けたstageのセルは`NOT_RECEIVED`。

## 4. SOURCE・candidate・lawの照合条件

- 照合するもの：C PLAN `c596e88d`、C実行SOURCE `1a87e533`、B base task（`3b742fff` static plan）、`build_menu/request_context`のfull identity。
- lawの扱い：
  - B `law_binding`はBのrequest（`b.LAW`）を表す。
  - Cの宣言とsamplerは`c1_original_law_binding`と`law_context`に残す。
  - 対応付けの根拠は、両commitの`rte.py`が同一blobであること（`law_correspondence`）。
  - ideal lawと実samplerの距離は未資格。
- 次の不一致は拒否する：
  - candidate、δ・prefix・K・R、allocation、compiler・profile、測定符号・単位
  - cohort、n_paths、execution SOURCE、result path、frozen commit
  - 価格値とevaluationの不一致
  - forced cohort
  - qualified=trueのmodel、既知扱いされたnull不確かさ

## 5. 価格を使える場合・使えない場合

| 状況 | 扱い |
|---|---|
| `qualified_expectation`（既定） | C1のPR価格はすべて`UNQUALIFIED_C`。総費用の正式価格にならない |
| 明示した`finite_cohort_scenario`＋direct cohortの選択 | direct cohortはscenarioとしてexact会計まで通る（mockで確認）。全law期待価格ではない |
| model予測 | qualified=falseのため使わない。採用閾値と必要精度は別途の科学判断 |
| C1外のq/R（m12長q、R8192等） | recordなし。`NEW_R_PRICE_NOT_ACQUIRED`。小qから外挿しない |
| 固定R provider | H4 R4/H6 R16だけ。新Rの値はtransplant guardで停止する |

同じ価格recordが既にある行に対して、`run_missing`がC providerを再び呼ぶことはない。sampling・compile・fitへのfallbackもない。

## 6. Aの将来N結果との接続条件

- N recordは`law_binding=r.request_context(menu,row)['law_binding']`、`N_comparison_policy=R_AWARE_EMPIRICAL_ACTION_COUNT_V1`で、C1 recordと同じcandidate IDとidentityを持つ。
- C1の新4セル（H4 m3/m4 R8、H6 m4/m5 R32）のNは現在欠測。同じ点で実N×Cを見るには、A側の別の科学実行が必要になる。
- A0のnormalizationは診断statusとしてだけ表示し、Nやshot係数・価格の乗数にはしない。
- A0とC1の共通点は、H6 m4 R16/R32の2点だけ。H4 R8の2点はA0の24件へ自動追加しない。

## 7. 実G/R比較を始めるのに足りない科学的資格

- 対象全候補のN（同じallowance policyによるbaselineの再解析を含む）。
- 長q・大Rの価格の有限検証（C2等）。
- modelの採用基準。
- 全law期待価格とrare寄与、actual samplerとideal lawの距離。
- cross-R・cross-round・shared calibrationを含む同時不確かさ。
- 全RPE成功（prior/alias/decoder/全round/failure/implementation-law）の資格。

これらが揃い、ユーザーが実費用評価を明示的に許可するまで、`r.evaluate_menu`／`b.run_evaluation`を実データで呼ばない。
