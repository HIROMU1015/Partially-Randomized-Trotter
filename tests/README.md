# tests の役割

研究B [単一複素signalのshot加重資源評価：技術設計・将来scope](../docs/research/research_b_outer_shot_weighted_resource_design_20261025.md)。設計のみ、実shot数／G／新順位の評価なし。新人工test `tests/management/test_b2d_shot_design.py` と既存scalar testの計10件PASS。将来案は6候補×両T×2精度、48 shot要求・144条件付き資源行。計算SOURCE／launcherは未作成、実行は別途明示許可後のみ。

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

`management/test_pr_dictionary_f2_attribution.py` は、保存F2寄与分析の分類保存則・共有metadata・joint origin・欠落／重複拒否・非零の記述分解を人工recordだけで検査する。新しい科学回路は生成しない。[結果とSOURCE対応](../docs/research/pr_algorithm_improvement_phase2d_f2_attribution_20261024.md)。
