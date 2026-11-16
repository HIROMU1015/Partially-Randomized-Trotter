# 利用者・C・統合担当・D2へ：PR-Pauliの広域schedule探索（round05の準備）

正本は [REPORT](REPORT.md)。B2（Claude Code）が作成し、確認は作成者自身による。新規の科学計算は0件。

## 次の実探索の開始・再開（承認後）

1. `configs/rpe_method_optimizer/rounds/h4_pr_pauli_schedule_wide_round05.json` の `science_approved` を `true` にする。現在はfalseで、実行は未承認。
2. 次のコマンドで開始する。同じコマンドで再開でき、完了済みのQ3・信号・価格・D2のREADY記録は再利用される。
   - κ、L_D、各roundのshot数の指定は不要。

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/b2-rpe-split-preparation-20261116
PYTHONDONTWRITEBYTECODE=1 \
/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python -B \
  scripts/run_rpe_schedule_search.py --problem configs/rpe_method_optimizer/h4_df10_problem.json \
  --round configs/rpe_method_optimizer/rounds/h4_pr_pauli_schedule_wide_round05.json \
  --run-dir docs/research/rpe_method_optimizer_h4_20261116/kappa_wide_round05/run01
```

- **設定の中身**
  - 候補の選択順：`search_overrides.schedule_wide`（梯子4倍、gap 2倍、細分化33/32、細分化2回ごとに探索1回、窓3/2、κ≤65536）。
  - shot配分：`coarse_gain_fraction` 1/100、`lookahead_mode` HYBRID。
  - 停止規則と、必要な科学計算の範囲（`scope`）。
  - Q3の予算：1,500件。
- **初回に評価するscheduleの見込み**：`first_decision/FIRST_DECISION.csv`（メタデータのみで確認済み）。
- **D2の入口が書き込む先**：B2専用のA checkout `b2-pauli-auto-workspace-20261116`（branch `codex/b2-pauli-auto-workspace-20261116`）。D2の入口はここでcommit・pushする。
- **呼出し側branchへのcommit**：D2のAPIは、比較用の成果物を呼出し側（当branch）にもcommitする。

## Cが参照できる固定path（このHANDOFFを含むcommit、`docs/research/rpe_method_optimizer_h4_20261116/kappa_wide_round05/`）

| 内容 | path |
|---|---|
| 全完了Q3の照合台帳、2種類の最安 | `ledger/LEDGER.csv`、`ledger/LEDGER_SUMMARY.json` |
| 旧規則・新規則の再生比較 | `replay/REPLAY_SUMMARY.json`、`replay/REPLAY_REACH.csv` |
| 初回選択（メタデータ） | `first_decision/FIRST_DECISION.csv`、`first_decision/KAPPA_STATE.json` |
| 受入記録 | `acceptance/`、`checks/SCHEDULE_SEARCH_CHECKS.json` |

- **構成別の最安の扱い**：構成別の「照合済みの全評価での最安」は、規則が採択した最安と分けて記載している。
  - LD5 κ=2：規則は501,233,190.43、全評価では500,962,225.57（round04のjob `9d10cc9f815edeeefdd7`、先読みで採択外）。
  - PR-Pauli全体：410,199,237.00で変化なし。

## D2へ

- **merge**：`98b40e439` を当branchへmergeした（`ba35de77b`）。
  - `run_rpe_optimizer_q3_job.common_metadata` は、`ENTRY_SOURCES` がないB view（`bd42dd1e9`）でも `ENTRY_SOURCE` で照合するように変更した。
  - D2のtest 52件は、merge後もOK。
- **run02の写しの再開**：当branchのコードで再開し、新規の科学stage 0件、Q3 0件を確認した。
  - 同じ設計の保存READY記録（LD4 κ=2）は、`complete` を呼ばずに再利用する。
  - PR-DF LD6の記録は、PR-Pauli LD6には使わない。

## ロードマップへの差分（本体は統合担当が更新）

- PR-Pauliの探索規則を、κ=2中心・範囲[1/2, 8]から、全κ範囲のschedule単位の探索（SCHEDULE_WIDE_V1）へ移す準備ができた。
  - 価格下界による証明付きの打ち切り、HYBRID並列、未準備の分割に対するD2の自動準備を含む。
  - 実行は未承認。
- 保存結果の再生では、新規則のほうが少ない答え数で437.9M・414.3Mの水準に到達した。410.2Mには未到達。旧規則が作った保存データ上での比較という制約がある。
- LD5の遠方schedule（κ=97〜6589）は、1 shotずつの価格下界が71M〜171Mで、保存最安410.2Mより低い。このため評価の対象に残る。ここは未評価で、PR-Pauliの最安は更新されていない。
