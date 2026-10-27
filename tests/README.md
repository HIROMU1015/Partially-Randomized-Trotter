# tests の役割

`tests/` は、ライブラリAPI、数値恒等式、成果物schema、ガード条件の回帰検査を置く。
基本的に `src/trotterlib/<name>.py`、`scripts/run_<name>.py`、
`tests/test_<name>.py`、`docs/<name>.md`、`artifacts/<name>/`を一組として読む。

テスト通過は、実装が期待した規約を満たすことを示す。一方で、次を単独では意味しない。

- 大きな系でも同じ近似精度になること
- 実量子backendやnoise下で成立すること
- 未検証のパラメータ範囲への外挿が正しいこと
- 最終的な総コストや手法間の優位性が確定したこと

研究上の証拠statusは[`../VALIDATION_STATUS.md`](../VALIDATION_STATUS.md)と
[`../artifacts/validation_manifest.json`](../artifacts/validation_manifest.json)を参照する。

最新のRPE長round検証は`test_rpe_target_round_horizon_validation.py`と
`test_rpe_delta_round_schedule_validation.py`、`test_rpe_delta_compiled_cost_validation.py`を、
対応する検証文書・artifactと一組で読む。最後の検証は中央RTEブロックだけを扱い、
Hadamard 1 shot全体や最終総コストの検証ではない。

P-A v1 blind transferは`test_research_direction_joint_synthesis_blind_validation.py`で、固定source/artifact
hash、事前登録gate、checkpointと最終artifactのtamper検出を確認する。H5 physical transferと
H4 optimization-level-2 compiler transferは両方とも完了している。
`test_research_direction_joint_synthesis_formalization.py`では、DPと全列挙の一致、syntheticな非退化分割、
完成済み54 recordの一区間退化・order 0 coverage、artifact tamperとscope guardを検査する。
`test_research_direction_joint_synthesis_mechanism_validation.py`では、計算前expected-task manifest、
全30 taskのorder-2 coverage、明示的一区間baselineとの0 split・0 plan差・0 RZ改善、固定停止判断、
artifact tamperとscope guardを検査する。
`test_research_direction_geometry_tracking_breakdown.py`では、16 taskのcompile-before manifestとsource hash、
訂正済み先行P-Cとのtraining再現、追跡prefix不変、stretch側blind予測破れ、固定停止判断、
artifact tamperとscope guardを検査する。
`test_research_direction_pd_fair_comparison.py`では、共通時間のexpected manifest、source freeze、finite K2/K4 cost、配分不能点のinfeasible記録、S1後の強制停止とscope guardを検査する。
`test_research_direction_pd_s1_posthoc.py`では、固定S1 fingerprint、一次Case Bの保存、主baselineの事後解釈、B1aのm_D診断、構成内訳、artifact改変拒否を検査する。

`test_pr2_s0_s1_validation.py`では、generation-prefix adapter、snapshot改ざん検出、corrected finite-RTE
mean、$\mathcal B^2$ shot式、q=1/8 controlled wrapperのRe/Im規約、stage gate、非上書きを検査する。

`test_pr2_new_series_validation.py`では、固定amendment/input hash、development二回load、held-out非開封、
raw/internal tamper、rank 3/6/9のpartition・sampling sign・確率和・再構成、counter、fingerprint、
非上書き、V4非承認guardを検査する。

## P1 selected integration regression

`test_df_research_pipeline.py` keeps the original 46 study-wiring/boundary case topics and adapts the three global-facade/output assumptions to the preserved eager initializer.
It adds only callback/policy/estimand/CLI-output failure coverage.
Actual current run: **78 PASS / 0 FAIL / 1 SKIP** (79 collected).
The SKIP is real legacy atomic publication because the eager initializer requires unavailable `matplotlib`.
Artificial-writer directory/collision tests passed separately and do not prove real atomic publication.
The old integer6/executor7 cases were not run: normal imports for both modules fail on that dependency.
No root package bypass or scientific dependency installation was used.
See [scope/reproduction](../src/trottertracks/resource_applicability/README.md).
Historical [P1 62 PASS](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/af54aeb7ebac042a3ca02379d61516ac635feef9/docs/integration/p1_df_pipeline_h4_20261027) and [adoption-review 19 PASS](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/76fec27f45b64879cdf5a86a32772ba78bd0a038/inventory/integration/p1_adoption_review_20261027) are distinct code-version records and are not added to this run.
Do not run scientific suites, old runners, sampling or build/compile under this technical integration scope.


## P1/P2 dependency, writer and H4 qualification validation (2026-10-28)

See `docs/integration/p1_p2_validation_and_h4_equivalence_20261028/REPORT.md`, `scripts/integration/run_p1_p2_technical_validation.py`, `tests/test_pipeline_real_writer_validation.py` and `tests/test_pipeline_eager_import_validation.py`. Real normal-import/writer tests pass; H4 numerical comparison is SOURCE_OR_INPUT_UNQUALIFIED. No scientific launch mode or old evidence is changed.
