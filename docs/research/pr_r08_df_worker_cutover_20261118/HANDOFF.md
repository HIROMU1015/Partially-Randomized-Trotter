# R08のPR-DF信号worker修正版への引継ぎ

GPUA100-gleap02のR08を同じround ID・探索範囲・200サイクル上限で継続する。
実行SOURCEは `4d4fc04a10fb07a5561bebc0ecf7333c69b808f8`。
旧controllerはサイクル18で新規投入を停止し、exit code 0で終了した。
今回の直前runで完了したQ3はPR-Pauliの128件（適合77、不適合51）。
直前runとそれ以前のrunを通常の保存結果照合に渡す。既存のQ3保留17件も引き継ぐ。

PR-DFの8候補は旧workerが固定providerのimportで終了し、contextの作成前だったことを旧ログ・固定request・出力状態で確認した。この8候補だけを既存探索の準備対象に戻した。元の失敗記録は旧runに保存されている。対象の候補キーと各固定locatorは [PREFLIGHT.json](PREFLIGHT.json)。

tmuxセッションは `h4-pr-r08-df-worker-fix`。同じrunを再開する際は、先に次で稼働を確認する。

```bash
tmux has-session -t '=h4-pr-r08-df-worker-fix'
```

セッション終了後にこのworktreeのルートから再開するコマンド。

```bash
bash docs/research/pr_r08_df_worker_cutover_20261118/START_OR_RESUME.sh
```

進捗は `run01/CONTROLLER.log`、`run01/DRIVER_PROGRESS.jsonl`、`run01/PROGRESS.jsonl`。PR-DF信号workerの段階ログは `run01/completion/candidates/<key>/df_signals/PROGRESS.jsonl`。
