# R08のPR-DF保存信号照合修正版への切替

サーバーは GPUA100-gleap02。修正版の固定実装は `f79708a7126ecbd827747bd261289d93496d9fa4`。
旧controllerへ新規投入停止を要求し、旧executorとcontrollerの所有lock解放を照合した。
旧第17サイクルまでのQ3 247件の正常終了とreceiptを保全し、通常のprior ingestへ渡す。
旧探索範囲・Q3予算・200サイクル上限と17件の旧配分保留を継承する。[PREFLIGHT](PREFLIGHT.json)

旧版の `df-plan` で保存信号の重複競合として保留された14件を再検討できるようにした。
旧記録に科学worker開始がない候補に限る。旧保留とログを元runに保存し、修正版の独立出力先で照合する。
保存信号の実質的な競合や未解決の科学attemptには既存の保留条件を適用する。
旧runの圧縮DPキャッシュは同じsource-keyで参照する。新しい科学条件や探索予算を追加していない。

tmuxは `h4-pr-r08-df-signal-fix`。再開前に `tmux has-session -t '=h4-pr-r08-df-signal-fix'` で確認する。
同じworktree・run directoryでの通常起動と再開は `bash docs/research/pr_r08_df_signal_cutover_20261118/START_OR_RESUME.sh`。
進捗は `run01/PROGRESS.jsonl`、`run01/DRIVER_PROGRESS.jsonl`、`run01/CONTROLLER.log`。
