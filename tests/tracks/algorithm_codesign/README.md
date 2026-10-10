# Track B algorithm_codesign tests


## G4 focused fixtures

`test_g4_independent_certificate.py`（13 tests、stdlib independent exact arithmetic）と
`test_g4_matched_cts.py`（17 tests、off-domain x1/3 semantics/certificate mutants）。
新規synthesis/science signal/registered cost取得0、full suite未実行。
[G4 handoff](../../../docs/tracks/algorithm_codesign/g4_results_and_gpt_handoff_20261009.md)。


## G5 focused tests（2026-10-10）

[test_g5_fixed_dictionary_closure.py](test_g5_fixed_dictionary_closure.py)：14 stdlib unittest、off-domain x1/3とsynthetic costのみ。
頂点mean/coverage、E sqrt(C) price、digital bridge符号、CTS full mean/phase/support/費用/区間/confidenceを確認。
registered table/lawをtestsで開かず、source-bound local PASS。full suite・science/synthesis/LPなし。
[record](../../../artifacts/track_b_g5_fixed_dictionary_closure/2026-10-10/focused_tests.json)、
[scope](../../../docs/tracks/algorithm_codesign/g5_fixed_dictionary_closure_scope_20261010.md)、
[handoff](../../../docs/tracks/algorithm_codesign/g5_results_and_gpt_handoff_20261010.md)。mandatory STOP。


## G6 independent return aggregation tests（2026-10-10）

[test_g6_return_aggregation.py](test_g6_return_aggregation.py)：16 focused stdlib tests。
独立adjacent-pair deletion oracle、GF/挿入/非負/符号mean、mutation controls、dyadic proposal/weights。
独立に選ぶoff-domain形式値のみ、registered science table/cost reads0。新performance結果ではない。
[保存log・scope](../../../artifacts/track_b_g6_return_generator_audit/2026-10-10/)、
[一般証明](../../../docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md)。


## G7 focused tests（2026-10-10）

[test_g7_budget_and_control.py](test_g7_budget_and_control.py)：18 off-domain formal/digital/control/launch tests。
registered synthesis結果を先に取得せず、matrixは8×8 synthetic semantic確認のみ。fullsuiteなし。


## 2026-10-10 Track B G7：取得完了・mandatory STOP（最新追記）

[G7結果/GPT引継ぎ](../../../docs/tracks/algorithm_codesign/g7_results_and_gpt_handoff_20261010.md)。
`G7_LIMITED_IMPLEMENTATION_ECONOMICS_COMPLETE_AWAITING_GPT_REVIEW`。source ab2549f41b3546fb3940342a2162dd9ee93699c4、24 keys/8 rows、strict error PASS、retry0。
固定P5では登録3対照後にもconditional期待T減少、P3ではclosed-form対照が小さい。
hard shot cap・classical generation/angle acquisition・未指定provider costを併記。
新規性/主method/次stage未採択、G5閉鎖/G6原証拠とmarker保持、GPT判断へ戻す。


## Track B G8（2026-10-10、source preparation）

[G8 proof/contract](../../../docs/tracks/algorithm_codesign/g8_proof_contract_and_on_demand_scope_20261010.md)：採用GPT G7 review §11に基づく限定確認。
2 known development inputs/4 same production laws、finite-provider parameter、分離failure配分、
on-demand strict Rzと対称bounded cache。G7のstatus/point comparison/consumed markerを保持。
17 off-domain focused tests、source preparation時点のnew native acquisition0。一束後mandatory STOP。
[test_g8_on_demand_and_bounds.py](test_g8_on_demand_and_bounds.py)：source freeze前のstub/backend0確認のみ、fullsuiteなし。


- G8: 17 focused off-domain tests passed before source freeze; saved-output audit14 passed after one-shot. [Scope/results](../../../docs/tracks/algorithm_codesign/g8_results_and_gpt_handoff_20261010.md). No post-STOP acquisition/testing extension.


## Track B G9 source preparation（2026-10-10）

[Fixed proof/scope](../../../docs/tracks/algorithm_codesign/g9_p5_matched_native_contract_20261010.md)：GPT G8 review §14を採用。known P5/指定3-qubit provider、6 direct+5 helper診断、19 keys/新CTS1 key、23 focused tests。source固定後一束のみ、終了後mandatory STOP。

## G9 one-shot STOP（2026-10-10）

`G9_TECHNICAL_INCONCLUSIVE`：epsilon引数のFraction→mp.mpf変換で停止、native比較0 row。
新規helper attempts1 / 新規sequence取得0 / retry0。source・contract・過去941pathは不変。
23 focused testsと23 saved-output checksは準備/整合証拠で、native資源の科学結果ではない。
[G9 failure・GPT handoff](../../../docs/tracks/algorithm_codesign/g9_results_and_gpt_handoff_20261010.md)。mandatory STOP、次の研究判断はGPTへ返す。
