# 次roundへ：κ round01の停止時点と、V2での読込み

正本は [REPORT](REPORT.md)。SOURCE は `cbe2947ad`。B2（Claude Code）が作成し、確認は作成者自身による。

## 停止時点

- **round**：`H4_PR_PAULI_KAPPA_2o3_1_2_S2_DELTA_H_K2_R01`。2026-11-16 05:03 UTC、方針変更のため途中終了した（`INTERRUPTED_SEARCH_POLICY_CHANGE`）。候補枯渇ではない。
- **評価**：完了Q3 4,229件。規則が採用した3,334件、未採用の先読み895件。未起動は9件。UNKNOWNは0件。
- **PR-Pauliの最安**：410,470,823.86 RZ（LD5 κ×2、N=[48,45,42,39,24,27,14,20,16,16,8,7,2]、MSE/τ=0.99991）。保存最安541,743,905.57を更新した。
- **構成別の値**：[KAPPA_COMPARISON.csv](KAPPA_COMPARISON.csv)
- **全評価**：[run01/EVALUATIONS.csv](run01/EVALUATIONS.csv)
  - exact値はサーバー上の `run01/jobs/*/OUTCOME.json` にある。63 GBあり、gitには入れていない。
  - gitの記録には、OUTCOMEのsha256を残した。

## 次roundが読み込む方法

round fileに次を書く：

```json
"executor": {"prior_run_dirs": ["docs/research/rpe_method_optimizer_h4_20261116/kappa_round01/run01"]}
```

- **読み込む条件**：同じjob key（方式・構成・配分・経路kind・契約）で、同じ経路のもの（view・契約template・信号packet・binding）。identity照合を通ったものだけを、知識に読み込む。
- **再起動しないもの**：既知の配分は再起動しない。
- **確認済み**（[load_check](../priority_v2_next_round/load_check/ROUND_RESULT.json)）：読み込み4,229件、拒否0件、起動0件。9構成の最安は停止時と一致した。
- **雛形**：[h4_pr_pauli_priority_round02_template.json](../../../../configs/rpe_method_optimizer/rounds/h4_pr_pauli_priority_round02_template.json)
  - rule `METHOD_MINIMUM_PRIORITY_V2`、全L_D、budget 0。
  - 実行するには、承認後に `max_new_q3_per_method` と `science_approved: true` を設定する。

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/b2-rpe-method-optimizer-20261116
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python -B \
  scripts/run_rpe_method_optimizer.py --problem configs/rpe_method_optimizer/h4_df10_problem.json \
  --round configs/rpe_method_optimizer/rounds/h4_pr_pauli_priority_round02_template.json --run-dir <新しいrun dir>
```

## V2が次に要求するもの（budget 0での出力）

- **Q3**：LD5 κ×2のm5を1減らす（410,199,859.00）。その後m4・m3を減らし、LD5 κ×1／κ×2/3を探索する。
- **入力要求**
  - κ案：LD5 κ×4、LD3 κ×4/9、LD2 κ×4/9。初期候補は近いκの適合配分。
  - 新L_D：4・6・7・8・9（P0準備要求）。LD1は、登録済みで入力不足の構成。
  - 価格・信号・経路がそろうまで、Q3枠は使わない。

## AのPR-Pauli候補準備入口（`76bb92184`）

[PREPARATION_CONNECTION](../PREPARATION_CONNECTION.md) を参照。

- **次roundでの動き**：全L_Dを列挙すると、未登録のL_DがAの入口へ送られる。
  - 保存LD1は、価格不足のまま候補として登録される。
  - LD4・6・7・8・9は、準備要求（P0）として残る。
- **κ案**：`search.prepare_kappa_proposals: true` にすると、同じ入口で構成を作る。作られるのはmetadataで、価格はD2のMODELで引用する。

## ロードマップへの差分（本体は統合担当が更新）

- PR-Pauliの最安：541.74M → 410.47M（LD5 κ×2）。
  - κを変えると、同じL_D内で最大23%下がった（LD2 κ×2/3）。
  - 停止時点の値で、収束値ではない。
- 探索規則：V1（構成ごとに下り切る）をV2へ置き換える案。保存ログの検証では、停止時最安に届くQ3が4,148件から218件になった。ログに条件付けられた比較である。
