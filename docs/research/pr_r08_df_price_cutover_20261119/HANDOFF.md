# R08のPR-DF価格照会修正版への引継ぎ

GPUA100-gleap02でR08を継続する。実行SOURCEは `e25179a74b5b2dad281ec2c73b0d26cb151425b6`。round ID、探索範囲、200サイクル上限、承認済みQ3予算は直前runの設定を引き継いだ。

[PREFLIGHT.json](PREFLIGHT.json) に旧runの停止状態、保存済みQ3件数、Q3保留を含む前runの固定参照と、切替時点の入力hashを記載した。旧候補・価格・信号の結果はそれぞれの元出力先に保全した。

[PRICE_RECHECK.json](PRICE_RECHECK.json) に旧版の複数価格モデル保留を保存証拠から再照会した結果を記載した。各軸で単価を返せたか、較正範囲内のモデルが競合したか、外挿の根拠が必要かを区別する。この再照会時の新規信号・compile・fit・Q3は0件。保留が残る候補は通常探索でも価格待ちとして扱う。

旧runはサイクル20で正常停止した。直前runのQ3はPR-Pauli 156件、PR-DF 98件を引き継いだ。再照会した4候補・104軸では、20軸で単価が得られ、54軸は出所選択待ち、30軸は外挿の追加根拠待ちだった。4候補とも全26軸価格は未成立。

サーバー上のtmuxセッションは `h4-pr-r08-df-price-fix`。同じrunを再開するときは、先にセッションの有無を確認する。

```bash
tmux has-session -t '=h4-pr-r08-df-price-fix'
```

終了後にこのworktreeのルートで再開する場合の入口。

```bash
bash docs/research/pr_r08_df_price_cutover_20261119/START_OR_RESUME.sh
```

進捗は `run01/CONTROLLER.log`、`run01/DRIVER_PROGRESS.jsonl`、`run01/PROGRESS.jsonl`。準備の詳細は `run01/completion/PROGRESS.jsonl`。
