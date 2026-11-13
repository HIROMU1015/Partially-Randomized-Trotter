# SOURCE・再利用・人工回帰の確認範囲

2026-11-13 JST。基点`952ce0113a27b179e4e6b889db124bdf37742fa1`、既存branch
`codex/d-compiled-residual-c-minus-s-20261113`を再利用。関連branchを確認したが、新候補探索器やN solverを作らず、Dの価格設計の継続として同branchに資料/人工testだけを追加する。

## 固定SOURCE

| commit | 実際に読解・確認した内容 |
|---|---|
| C `9d95d3ea544e1fa1d0e4710c1ca24d8c5776469f` | `h4_structural_schedule.py`: prepareの固定menu/EXPECTED_R・ScientificBackend、`h4_structural_kernel.py`: Model/Wire・固定component/basis/preparationとboundary。全26 E[S]の既存HANDOFF。保存値の数値再解析なし |
| C `37cbc331b6cb9d38e83584f8e5b007da173bc5cd` | `docs/research/structural_feature_expectation_design_20261113/REPORT.md`, `HANDOFF.md`、`src/trottertracks/structural_feature_means.py`: k2_linear_counts/k2_basis_counts/k2_basis_kernel。SOURCEを読んだだけで実確率を与えていない |
| D `39ddd59edf54a3a901e5302207b2765ddbd03fbb` | `docs/research/d_p2_direct_pilot_20261113/HANDOFF.md`等の既存記述とrunnerの範囲。raw/RESULT/PATHSから新しい統計・fitなし |
| D `952ce0113a27b179e4e6b889db124bdf37742fa1` | `docs/research/d_compiled_price_methods_20261113/REPORT.md`, `C_HANDOFF.md`, `MINIMAL_SCIENCE_PLAN.md`。未証明tailとm12 UNKNOWNを継承 |
| B `cdf6a80671b2d0bd58983fbbbaf7334f58dc8422` | `docs/research/h4_mean_price_validation_strategy_20261113/REPORT.md`, `PRICE_VALIDATION_PLAN.md`, `HANDOFF.md`, `FINITE_SCIENTIFIC_PLAN.md`。ΣNb/同時区間/N未確定/長時間domainの既存設計 |
| C1 `1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7` | `src/trottertracks/r_price_phase1_science.py`: Backend.context/system-only cache、prepare固定L_D、S headの固定刻み/recipe、price_requests。実contextは起動していない |
| carrier `155529e20a8fac39e00786c40b57830b71203c4f`、現保存版 | `scripts/h46_long_time_native.py`: trial_seed/sample_requests/compose。signed piece時間・preparation照合、scalar一回、seedのcandidate/outer/piece binding |
| fast pipeline `5a67cea7705e80ff295306606611d774724ae2e2`由来、基点保存版 | `src/trottertracks/compiled_price_fast.py`: structural_pair/make_backend_class、`efficient_controlled_zz.py`。元contextを継承する適用制約、H4 wire固定・両軸個別compile・cache/fingerprintを再確認 |

C/Bの追加SOURCEは`git show <commit>:<path>`で閲覧した。B指定SOURCEの
pr_rpe_rmse_controller、rpe_r_selection、compiled_residual、bv2_mean_priceは基点保存版とbyte同一を確認した。
各研究のraw/input/KEEP/grant/ledger/marker/STOPを上書きしない。今回新しいscientific inputのロードなし。

## 既存APIを再利用する範囲

- `pf_pr_rpe_pipeline.identity/structure/measurement`：canonicalな科学identity・signed配分・axis単位。
- `pr_rpe_rmse_controller.assess/cost/select`、`pr_rpe_rmse_evidence.plan`：外部V2 schedule・全体N/RMSE資格・quantity・law/cohort・欠測・whole-schedule比較。新しい実N/C/Gを与えていない。
- `rpe_r_selection._bounds/evaluate_menu`：完全R menuの供給同時区間とpoint価格の比較。部分未知/bound-onlyを既に受けるとは主張しない。
- `h4_df10_rpe_connection.exact_scalar/exact_value`：巨大整数Nとexact有理数の算術。新しい算術layerを作らない。
- `pf_pr_rpe_pipeline.joint_uncertainty`：2armの供給covariance/区間経路。float64重み/和の限界を維持。
- `bv2_mean_price.V2MeanPriceLookup`、`compiled_residual`、`pr_rpe_rmse_integration`：保存価格/physical scheduleとのbridgeと未資格・nullの保持。新しいautomatic provider/実行launcherなし。

bound-only/partial-unknown/signed-price-differenceの汎用選別をproductionに新実装していない。
今回はその最小接続仕様と人工式確認を渡す。Bの既存controller/探索担当と役割を重複させない。

## 追加テストと再現

追加は[`tests/test_rpe_price_search_design.py`](../../../tests/test_rpe_price_search_design.py)だけ。
人工N/C/区間により、部分下界、signed差分と依存性、同一quantityの相殺条件、τ-regret、N重み、N集合、巨大整数、L_D/lowering/quantity不一致、欠測、逐次failure会計、paired covariance/control variateの限界を確認した。

```bash
PYTHONPATH=src:tests:scripts /home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -m pytest -q --disable-warnings tests/test_rpe_price_search_design.py tests/check_pr_rpe_rmse_controller.py tests/check_rpe_r_selection.py tests/check_pr_rpe_rmse_evidence.py
```

[TESTS.txt](TESTS.txt)：新12＋既存controller17/R-menu12/evidence13＝**54 PASS、7.03秒**。
作者自身の人工・回帰確認であり、独立科学レビューや実価格精度の検証ではない。
実行では人工fixtureの既存generate/assess/cost等を使った。分子の候補生成・shot計算・回路生成・compiler・保存値再解析は起動していない。既存regressionの点Gは人工値だけである。

科学run用adapter、price range証明、平均推定model、bound-only本番controllerは今回追加しない。
これらの必要性は外部menuと比較目的が届いた後に決める。公開後mandatory STOP。
