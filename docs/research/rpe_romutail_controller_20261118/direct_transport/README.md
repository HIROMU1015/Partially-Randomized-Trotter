# RomuTailからの直接配送と実機人工検収

実機3台の通常入口検収は `REAL_THREE_ACCEPTANCE.json` が正本で、13件すべてを一度だけ採用した。内訳は信号4件、共有compile 1件、共有audit 1件、人工Q3 7件。RomuTailに7件、server2に4件、WindowsのSSH接続先から入るlocal0-wslに2件を割り当てた。server1は無効のまま接続していない。controller/通信は `89b3da81a0905b41238dea06b0f4f993d28c86d6`、実行するworkerと人工leafは `d887abdcadf47a969aa2f69c1dd3bdc88c20f1c2` に固定した。

`REAL_THREE_FAULT_CHECKS.json` は、途中参加、信号とcompileの並行、別ホストaudit、途中READYからQ3への復帰、転送前切断・同じattemptの回収、Pause中の回収、復帰、完了後Resumeの追加起動0を記録する。切断は専用relay内でlocal0-wslの通信失敗を人工的に起こした。独立した読み取り専用mailbox照会で、WSL側の仕事と保存が継続したことを確認した。物理ホストやネットワークを停止した検査ではない。異なるホストの時計だけで並行性や片道転送時間を判定していない。

`REAL_THREE_SINGLE_HOST_REFERENCE.json` は、保存された同一人工Q3 REQUEST 7件を固定d887の既存人工leafへ逐次入力した参照との比較で、MSE上下界の有理数と判定が全件一致した。実H4・実compile・実Q3は0件。この人工leafはDPを使わず、この検査は数値DP cacheの検収ではない。

直接配送は既存 `scripts/run_rpe_q3_relay.py` と `DistributedExecutor` / `JobStore` / mailboxを使う。RomuTail上のcontrollerはloopbackへbindし、relayの `external_tunnel=true` で直接照会する。workerごとの既存 `ssh_command`、`transport="ssh-wsl"`、`distribution`、受付ファイルを指定できる。Windows経由のWSLでは、JSON入力をSCPで一時ファイルに転送し、WSLプロセスへ標準入出力を明示して渡す。返却時に一時入力を削除する。`WSL_FILE_INPUT_ROUNDTRIP.json` は225,280バイトの人工CAS put/getと状態照会の実機一致を記録する。認証設定は既存SSH設定を利用する。

通常利用者の操作は上位の [HANDOFF](../HANDOFF.md) を参照する。専用人工fixtureの再現では、WindowsクライアントがcontrollerのStartを担当し、先に次の検収補助だけを起動する。

```bash
task_python='/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python'
task_source='/home/abe/Project/Partially Randomized Trotter/.rpe-central/releases/89b3da81a'
task_fixture='/home/abe/Project/Partially Randomized Trotter/.rpe-central/acceptance/real-three-v2-d887abdca'
"$task_python" -B "$task_source/scripts/check_rpe_romutail_distributed.py" --exercise "$task_fixture"
```

完了済みのこのfixtureを再実行する手順ではない。新しい検収には新しい所有ディレクトリ・profile・spoolを用意する。`--check` は保存証拠の集計、`--verify-resume` は完了後の同じrunでlaunch・採用回数・成果hashが増えないことの照合に使う。

`FINAL_WSL_PARTICIPATION.json` により、実workerの受付OFFと設定hashが前後で一致することを確認した。人工fixtureの受付も検収後はOFF。`WSL_CLONE_METADATA_CHECK.json` は固定sourceから既存準備workerのshared/sparse cloneが実際に成立したmetadata検査で、科学leafの起動は0件。

最終controller `6da8554e0040f98578667973e8802c680f87982e` の通常Windows Start・実行中Stop・Resumeは `ACTIVE_STOP_FINAL.json` に記録した。workerは同じd887。2件目の人工Q3が実行中に停止要求を受け、元の仕事が完了・保存・採用されてから正常終了した。StopとResumeの終了値はともに0で、Resume後も2件のattempt・成果hash・LAUNCH・採用回数が一致し、追加起動は0件。既存の停止処理が記録した2件の `KeyboardInterrupt` は保全し、Resume時に追加のruntime errorがないことも照合した。

初回の準備検収で見つけた設定locatorとexecutor寿命の不具合は修正後の新しいfixtureで検査した。初回の失敗したqueueと成果は元の場所に保全し、`SETUP_DIAGNOSTIC.json` に区別して記録した。実機人工検収と参照比較はdirect_transport担当Codex、WindowsのStart/Resumeは主担当Codex、クライアント再送検査はlocal0_client担当Codexが実施した。
