# R08：候補を保留して独立探索を継続

host `GPUA100-gleap02`、worktree `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-pr-candidate-isolation-20261117`、branch `codex/b2-rpe-pr-candidate-isolation-20261117`。

code SOURCE `07b9c1ce826ba67e7e8237b1d1c781a2e1357e6d`。roundは`H4_PR_BOTH_MULTIWIDTH_CONTINUITY_R08_20261117`、runは`docs/research/h4_pr_candidate_continuity_20261117/run01`、tmuxは`h4-pr-width-r08`。入力commitは起動時の`CONTROLLER_INVOCATIONS.jsonl`に記録する。

まず稼働と段階前進を確認する。

```bash
tmux has-session -t h4-pr-width-r08
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-pr-candidate-isolation-20261117
tail -n 8 docs/research/h4_pr_candidate_continuity_20261117/run01/CONTROLLER_INVOCATIONS.jsonl
tail -n 8 docs/research/h4_pr_candidate_continuity_20261117/run01/DRIVER_PROGRESS.jsonl
tail -n 8 docs/research/h4_pr_candidate_continuity_20261117/run01/PROGRESS.jsonl
```

controller/所有workerがいないことを確認してから、同runを再開する場合：

```bash
bash /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-pr-candidate-isolation-20261117/docs/research/h4_pr_candidate_continuity_20261117/START_OR_RESUME.sh
```

元の通常`run_rpe_schedule_search.py`を`--batch 64 --max-cycles 198`で起動する。R07のcycle2から元のcycle200まで。queue20 slot、DP3 worker、m5/m11 prefix cacheを維持し、worker生存だけで正常進捗と判定しない。flockで二重controllerを拒否し、自己retry loopを追加していない。

## 保留の読み方

- `ROUND_RESULT.json.methods.<method>.execution_holds`：configurationごとの旧job照合待ち。保存適合点は保持し、未解決のjobを再計算しない。
- `input_requests.kind=SAVED_JOB_REVIEW_REQUIRED`：科学取得とは別の照合待ち。既存guardで完了結果を受入可能になった場合のみ解除する。
- `CANDIDATE_HELD`/`CANDIDATE_HOLD_RESOLVED`：保留・解除log。old SOURCEを現在と同一と決め付けない。
- `LATE_READY_INSTALLED_DURING_PREPARATION`：ほかの準備が続く間の新READY投入。同一queueと採択済み総予算へ戻る。
- 数値UNKNOWN、SOURCEや共有入力の異常、保存不能などの旧停止条件は維持する。重大STOPを単純再開で消さない。

R07の16 READYと5保留、完了Q3、registry、checkpointは出所を付けて再利用し、元R07を編集しない。過去SOURCEの一部は同じサーバーのGit object/rawからのみ取得できるため、公開remoteだけでの再現とは区別する。

終了時は`ROUND_RESULT.json`、`CONFIRMED_MINIMA.json`、`KAPPA_STATE.json`、jobごとのSOURCE/OUTCOME/終了記録から、元の要求どおり方式×要求幅の最安・配分・MSE・条件付き価格と未取得/保留領域を回収する。大域最適、law期待compiled価格の認定、formal G/winnerとは区別する。計算途中のチャット終了後も同じrunを引き継ぐ。
