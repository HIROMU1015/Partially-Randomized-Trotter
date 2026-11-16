# 人工検収fixture

`scripts/check_rpe_distributed_q3.py` は既存の
`tests/check_pr_width_search.py` の人工producerを再利用し、通常の
`run_rpe_schedule_search.main()` から探索を動かす。H4入力・信号・回路は使わない。
初回公開のfixture準備はmetadataだけで、workerは起動しない。

```bash
python3 -B scripts/check_rpe_distributed_q3.py --prepare /tmp/rpe-q3-acceptance-NEW \
  --port 8765 --local-slots 1 --total-slots 3
```

生成物は `PROBLEM.json`、`ROUND.json`、各ホストの `PROFILE_*.json`。
サーバー1・ローカルでは共通実装SHAを取得し、profileの `root`、
`interpreter`、`work_dir`、`prefix_cache_directory` をそのホスト上の専用pathへ直す。
各ホストは独立したspoolを持つ。共有ファイルシステムは不要。

```bash
# サーバー2: 通常schedule CLIを呼ぶ人工wrapper
python3 -B scripts/check_rpe_distributed_q3.py \
  --controller /tmp/rpe-q3-acceptance-NEW --port 8765 --local-slots 1

# サーバー1・ローカル: 許可済みSSH経路でcontrollerのloopbackへ接続
# SSHの方向・接続aliasは担当者の実測結果をHANDOFFへ記載する。
python3 -B scripts/run_rpe_q3_worker.py \
  --profile PROFILE_server1.json --controller http://127.0.0.1:8765

# 回収後
python3 -B scripts/check_rpe_distributed_q3.py --check /tmp/rpe-q3-acceptance-NEW
```

2方式・L_D=1,2・複数幅・最終roundの2段shotを扱い、初回Q3完了後に人工価格を公開して
待機幅をREADYへ戻す。数値leafはexact有理数の人工MSE。
人工負荷の実時間・速度倍率は実H4の性能と区別する。

`--acceptance DIR` は同じOS上の別profile/spoolを使う模擬1/2/3ホスト検収。
実サーバー間検収の代わりにはならない。実装と検収結果はREPORTを参照。

実機2台を待つ専用人工runの準備例:

```bash
python3 -B scripts/check_rpe_distributed_q3.py --prepare /tmp/rpe-q3-real-two-hosts-NEW \
  --port 8765 --local-slots 1 --total-slots 2 --sleep-seconds 1.0 \
  --required-hosts server2 server1
```

全指定hostがpoll登録するまで新規割当を待つ。接続方向はlocal0→qg1/qg2のみで、
サーバー間直接SSHを仮定しない。local0からのSSH mailbox relayのコマンドは
親の[HANDOFF](../HANDOFF.md)を参照する。
