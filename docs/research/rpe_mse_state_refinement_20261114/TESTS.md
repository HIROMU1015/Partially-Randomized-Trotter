# 技術検証記録

2026-11-14 JST。primary agent自身による人工・回帰検証であり、独立科学レビューではない。Pythonは`/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B`（3.11.0rc1）。

最終コード：**57件PASS、5.532秒**。[全test名と結果](TEST_RUN.txt)。

- 新`check_rpe_deterministic_refinement.RefinementTests`：12件。
- 既存`check_rpe_deterministic_risk.DeterministicRiskTests`：13件。
- `check_pr_rpe_rmse_controller.ControllerTests`：17件。
- `check_rpe_shot_allocation.AllocationTests`：MCを実行するtest_02とtest_10を除く8件。
- `check_rpe_decoder_fast.FastDecoderTests`：以下の7件。

```text
test_small_q_exhaustive_phase_grid
test_exact_ties_preserve_both_different_reference_rules
test_latest_published_B_statistics_blob_is_identical
test_wrap_midpoint_and_1e12_tie_neighbourhood
test_extreme_phases_signed_zero_and_rounding_boundaries
test_scalar_rule_does_not_enumerate_q_branches
test_invalid_inputs_and_missing_measurement_remain_rejected
```

最終suiteはS.simulate/S.simulate_prepared、W.diagnose/K.diagnose、p.run_missingをmockで禁止した。SavedArtifacts.readも保存実signal bundle/carrierのpathを拒否した。新旧testは人工count法則・人工価格だけを数値評価し、real H4への適用・旧結果の新再解析は0。旧MC/random入力を使うtestsは走らせていない。

新testの再現入口（人工のみ）：

```bash
/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B -m unittest discover -s tests -p 'check_rpe_deterministic_refinement.py' -v
```

`SYNTHETIC_COMPARISON.json`は同じtestのrefined_case/scalar_history_oracleとD.evaluateで作った人工表示例。新methodのサンプルであり、N資格・実H4の性能証拠ではない。数学的条件・未取得資格はHANDOFF参照。

元decoder/statistics/MC/risk interval/caller/shot/cost code、旧結果・入力・grant/ledger/marker/KEEP/STOPは変更していない。今回変更した数値moduleはrpe_deterministic_risk.pyの任意DP経路だけで、既定経路と新旧13round試験がPASSした。実H4の新MSE/必要N/compiled価格/Gは未計算。

最終SOURCE確認PASS：旧Dの関数は任意統合を足した_bound以外すべてAST一致、追加は_refine_statesだけ。固定H4 caller/入力/結果、元decoder/統計/会計・risk interval codeはGit差分なし。旧manifestの全entryを保持し、新文書のローカルリンクとgit diff --checkも確認した。この確認から科学数値を再評価していない。
