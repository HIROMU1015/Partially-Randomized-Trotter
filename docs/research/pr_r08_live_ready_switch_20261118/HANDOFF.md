# R08の修正版切替

旧budget回復runのcontrollerだけに一度SIGINTを送り、Pythonの `KeyboardInterrupt` 回収を使う。子workerは既存の終了・receipt書込を完了する。旧controller起動記録・終了記録・停止理由・所有lock・Q3 receiptを照合し、不明なjobがあれば引継ぎを停止する。旧SOURCEの人工検証でworker 2/2が終了し、終了コード `-2` と `KeyboardInterrupt` のSTOPを保存した。

新runは同じR08の条件、cycle上限、Q3 queue/DP、prefix cache、価格規則と先行runの結果を引き継ぐ。`live_ready_refill: true` を固定する。旧branchの一部private軌道ファイルがGitHubサイズ上限で拒否されたため、旧Git履歴・raw・停止metadataはサーバーの元worktreeで保全し、元commitを参照する。修正版のコードと新入力の公開branchは独立させる。新runでもprivate軌道はローカルに保全し、軸別receipt・freeze・holdout・モデルの登録に必要なファイルは通常どおり記録する。

host `GPUA100-gleap02`、worktree `h4-pr-ready-slot-refill-20261118`、tmux `h4-pr-r08-live-ready`。実行入口は `bash docs/research/pr_r08_live_ready_switch_20261118/START_OR_RESUME.sh`。同じ入口は所有lockがあるとexit 75。操作前に `tmux has-session -t '=h4-pr-r08-live-ready'` と `HANDOVER_STATUS.json` で動作を確認する。

状態確認は `HANDOVER.log`、新 `run02/CONTROLLER_PID.json`、`run02/PROGRESS.jsonl`、`run02/DRIVER_PROGRESS.jsonl`。`LIVE_READY_INSTALLED` と `Q3_WAIT_YIELD_FOR_READY` を実H4で確認したかは実際のログで区別する。
