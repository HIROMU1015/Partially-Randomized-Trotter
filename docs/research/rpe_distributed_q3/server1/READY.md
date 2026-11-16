# サーバー1受信口の引渡し（人工2台検収用）

共通workerを `89f51a4fef47ecd2364d22c4c5cc2e225df3f2ce` に固定して展開済み。
`--mailbox status` の実測は host_id=server1、空きslot=1、attempts=[]。
中央runからの依頼はまだ受信していない。新規H4計算0件。

- [PROFILE_server1.json](PROFILE_server1.json)：サーバー1上に実在する固定root・spool・interpreter。
- [LOCAL0_WORKER_ENTRY.json](LOCAL0_WORKER_ENTRY.json)：中継configのworkersに使う実path。
- [RELAY.local0.json](RELAY.local0.json)：ローカル0用config。controller remote_port=8763は共通既定値。
  サーバー2担当が人工runで別portを選んだ場合は、その実測値に変更してから起動する。
- [MAILBOX_READY.json](checks/MAILBOX_READY.json)：実行commandと応答。

## サーバー1：状態確認・受信再開

```bash
SERVER1_Q3_ROOT=/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/rpe-distributed-q3/common-89f51a4fef
SERVER1_Q3_PROFILE=/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/rpe-distributed-q3-server1-relay/docs/research/rpe_distributed_q3/server1/PROFILE_server1.json
python3 -B "$SERVER1_Q3_ROOT/scripts/run_rpe_q3_worker.py" --profile "$SERVER1_Q3_PROFILE" --mailbox status
```

mailbox方式は常駐poll daemon不要。ローカル0の`submit`がjob専用supervisorをdetachして起動する。
起動・再開はローカル0中継の`--start`。受信停止は中継の`--stop`。
停止しても受領済みjobとspoolは保持し、計算を続行する。再開時は同じprofile・spoolを使う。
サーバー1側にHTTP poll workerを重ねて同じspoolへ起動しない。
初回は同時Q3=1。DP workers/Q3は中央REQUESTのexecution_profileで別に指定される。
共通人工fixture既定値はDP workers=1であり、hostの128 CPUから自動選択しない。

## ローカル0：起動・状態・停止・再開

共通SHAとこの担当branchのファイルをローカル0へ取得し、共通checkoutで実行する。
`RELAY.local0.json`をそのdirectoryへコピーした場合のコマンド：

```bash
python3 -B scripts/run_rpe_q3_relay.py --config RELAY.local0.json --start
python3 -B scripts/run_rpe_q3_relay.py --config RELAY.local0.json --status
python3 -B scripts/run_rpe_q3_relay.py --config RELAY.local0.json --stop
# --statusのstopped_epochを確認後、同じconfigとstate_directoryで再開
python3 -B scripts/run_rpe_q3_relay.py --config RELAY.local0.json --start
```

`qg1`/`qg2`は利用者指定の既存alias。サーバー1からこれを実行済みとはしない。
server2人工controllerは同じ固定SHA、total_slots=2、required_hosts=[server2,server1]を使用する。
共通SHAが更新された場合は、中央manifestとサーバー1rootのHEADを一致させてから受信する。

## 未完了の実機検収

1. ローカル0でqg2転送とqg1 mailbox RPCを実行し、両端のhostname・source_shaを保存。
2. server2とserver1の異なるjob IDの実行区間の重なり、中央の両採用を照合。
3. 次候補がserver1へ届くまで確認。
4. 専用人工jobがserver1で完了した後、回収前に中継を停止。再開後の同一job IDの回収とLAUNCH総数1を照合。

上記のjob ID・中央receiptはローカル0/サーバー2側の実行後に追記する。
