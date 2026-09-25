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
