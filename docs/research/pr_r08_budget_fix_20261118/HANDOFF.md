# R08 budget修正・状態確認・再開

- 修正コードSOURCE：`2c2196bee51589b6765039691c1e7509169ef6af`。中核 `f3867a1f3`。
- host：`GPUA100-gleap02`
- worktree：`/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-pr-budget-continuity-20261118`
- branch：`codex/b2-rpe-pr-budget-continuity-20261118`
- round ID：`H4_PR_BOTH_MULTIWIDTH_CONTINUITY_R08_20261117`
- run：`docs/research/pr_r08_budget_fix_20261118/run02`
- tmux：`h4-pr-r08-budget`
- 入口：`START_OR_RESUME.sh` → **通常** `scripts/run_rpe_schedule_search.py`。
- 固定入力：RESUME_PLAN/PROBLEM/ROUND、run02/SOURCE、PRELAUNCH_ACCEPTANCE。

## 確認を先に行う

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-pr-budget-continuity-20261118
tmux list-panes -t h4-pr-r08-budget -F '#{pane_pid} #{pane_dead} #{pane_current_command}'
cat docs/research/pr_r08_budget_fix_20261118/run02/CONTROLLER_PID.json
tail -n 5 docs/research/pr_r08_budget_fix_20261118/run02/DRIVER_PROGRESS.jsonl
tail -n 5 docs/research/pr_r08_budget_fix_20261118/run02/PROGRESS.jsonl
tail -n 5 docs/research/pr_r08_budget_fix_20261118/run02/completion/PROGRESS.jsonl
```

異常時は `RUNTIME_ERRORS.jsonl`、CONTROLLER.log、CONTROLLER_STATUS.json。STARTEDだけでgate PASSとしない。実進捗はstageの完了・次stage開始とqueueのLAUNCH/EXIT/ADOPTを確認する。

## 同じrunの再開

**稼働中は起動しない。** wrapperは新runと旧R08の既存所有lockを保持する。二重起動はexit75。STOPを自動解除せず、失敗科学stageを再試行しない。

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-pr-budget-continuity-20261118
tmux new-session -d -s h4-pr-r08-budget 'bash /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-pr-budget-continuity-20261118/docs/research/pr_r08_budget_fix_20261118/START_OR_RESUME.sh'
```

wrapperが保存cycleから200までの残りを計算し、batch64、同じrun/固定入力で通常CLIを呼ぶ。元target/精度、要求幅のbinary64、L_D/κ探索、20 slot/DP3、prefix cache、worker SOURCEを維持する。新SOURCE/input commit/PID/実executableはCONTROLLER_INVOCATIONSとCONTROLLER_PIDへ残る。

## 旧結果・適用境界

旧固定結果：`f9e2ff4ecfdc41b254e316d5e74e42538a6bc518`、`docs/research/h4_pr_r08_recovery_20261117/run02/`。128job（適合92、不適合36）のREQUEST/OUTCOME/PROCESS_RETURNとKeyErrorの原traceback、旧STOPを参照できる。新しいrunはこれをprior_run_dirsで読み、元の再利用guardを使う。以前の316jobや更に前の保存run一覧も維持する。

lambda_source不一致5pointは個別保留。数値・SOURCE・保存障害のglobal保護は緩めない。予約拒否の `BUDGET_WAIT` は未実行要求を返すbatch制御であり、MSE判定や候補不適合ではない。実起動した先読みは採用されなくても予算へ計上する。

検証は追加5＋既存32＝37人工PASS、native追加5PASS。`checks/NORMAL_ENTRY.json` は人工通常入口・同じrunの再開追加0起動。実H4の逐次比較速度倍率は未測定。C/統合担当は最終ROUND_RESULT/CONFIRMED_MINIMAができた時に、確認済み集合・条件付き価格・規則採択/先読み・未評価/UNKNOWNを分けて取り込む。今回の技術修正を全探索完了や価格保証へ拡張しない。
