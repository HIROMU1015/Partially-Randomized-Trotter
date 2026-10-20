# Track B：採択済み限定Pilot 16,896件

今回のユーザー指示に基づく新namespaceの実装・人工受入を完了。実科学はまだ開始していない。これから実SOURCE commitとmanifestを固定し、同じ承認内容との一致をpreflightで検査してfresh scopeのone-shot実行へ進む。

- 原承認：user_approval.json。採択と具体化：adopted_contract.md。
- 人工受入：artificial_acceptance.json、artificial_normal_final_checker.json、semantic_mutations_final.json。
- Core不変性/境界：static_science_boundary_check.json。旧資料保全：preservation_before.json。
- SOURCE：src/trottertracks/algorithm_codesign/g6_w2c_b3_limited_pilot_live_v1/。
- focused tests：tests/tracks/algorithm_codesign/test_g6_w2c_b3_limited_pilot_live_v1.py。
- runner：python3 -m trottertracks.algorithm_codesign.g6_w2c_b3_limited_pilot_live_v1.authorized_runner（prepare/execute/inspect）。executeは一回限り。

人工試験は同じprimary Codexの技術検証であり、別reviewerの独立再現ではない。科学Core/Exact Adapter/Global Bundle/F01/Storage V2/旧checkerは不変。終端は必ずmandatory STOP、retry0/resumeなし。Full-N・追加pilotへの移行は承認されていない。
