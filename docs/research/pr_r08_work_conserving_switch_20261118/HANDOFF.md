# R08の待機候補補充・評価順への引継ぎ

最適化器は `1e5d63c326282382a311304e21d995ac09954003`。引継ぎrunnerを含む固定SOURCEは `PLAN.json` に記録する。新しい入力で `live_ready_refill` と `search_overrides.work_conserving_search` を有効にする。元R08の方式・幅・κ範囲・shot条件・精度・予算・worker profile・DP prefix cacheを継承する。

旧runの準備波が終了し、準備中候補が0件になった時点で、固定した旧controller PIDだけにSIGINTを1回送る。既存workerの終了と元receiptの保存を待つ。旧wrapperの状態・終了コード・STOP・所有lock・Q3終端receiptを照合したうえで旧結果を固定する。旧runとそれ以前の完了jobを既存の再利用経路に渡し、未確定の部分は元の保留条件で扱う。

稼働先は `GPUA100-gleap02`、worktreeは `h4-pr-ready-ingest-latency-20261118`。tmuxは `h4-pr-r08-work-conserving`。起動と同じ固定runの再開は同じ入口を使う。実行前にtmuxと `HANDOVER_STATUS.json` を確認する。

```bash
tmux has-session -t '=h4-pr-r08-work-conserving'
cat docs/research/pr_r08_work_conserving_switch_20261118/HANDOVER_STATUS.json
bash docs/research/pr_r08_work_conserving_switch_20261118/START_OR_RESUME.sh
```

出力はこのdirの `run02`。導入の確認は `PRELAUNCH_ACCEPTANCE.json`、`run02/SOURCE.json`、稼働の確認は `PROGRESS.jsonl`、`DRIVER_PROGRESS.jsonl` を用いる。 `Q3_STANDBY_REFILL` と `SEARCH_ADOPT.order_reason` は別々に観測する。評価時間を用いた順位は互換profileでの完了観測が蓄積してから使う。実際に観測した事象と速度改善の確認範囲を区別する。
