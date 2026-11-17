# 実H4 Q3一件のPBS移植検収・引継ぎ

**PASS：実Q3起動1・完了1・採用1。exact上下界と判定は元結果と一致。再開の追加処理0。**
[REPORT](REPORT.md)、[機械可読比較](results/EXACT_COMPARISON.json)、[再開](results/RESUME.json)。
元の科学判定は閾値不通過のまま。科学的GOや方式の優位性を追加した検収ではない。

## 固定版と正本

| 項目 | 値 |
|---|---|
| branch | `codex/rpe-pbs-h4-q3-portability-20261118` |
| 実行前caller/入力/設定commit | `5040a1813` |
| release | `romutail-pbs-integrated-925985b4e`（非currentの登録済み版を明示） |
| controller/PBS/worker SOURCE | `925985b4ea76a15f8a07c507179d4fa6c492eabd` |
| 科学Q3 SOURCE | `28d976b99cdf8329adcc1b79327e329e947ef774` |
| adapter / route | `349558b748a82f384a4589434a63ef9b5572032c` / `cbc471c913072de45c25a0ed22d51dff8a9b7034` |
| 元job／今回attempt | `98693b829295fc884330` / `18c2c7bc2d7b4a5581ef2543b1359e1b` |
| 中央正本run | `/home/abe/Project/Partially Randomized Trotter/.rpe-central/state/runs/pbs-h4-q3-portability-20261118/output` |
| 専用SERVICE/ROUND | この記録の`deployment/SERVICE.json`、`ROUND.json` |
| PBS / node | `267197.CPUcluster-PRISM` / `CPUcluster031`、F/exit0、worker EXITED |

原結果・STOP・既存中央runを上書きしていない。共有currentの切替は中央統合担当に集約したままである。
作業中に同担当側のcurrentは`romutail-price-fleet-e917165ba`、SOURCE
`e917165ba1f36b7627614cd25e88b5108d9b20b9`へ進んだ。PBS側はその切替を操作せず、今回のrun pinは925を維持した。
専用runは共有RUNS/CURRENT_TARGETへ新しい探索targetとして登録していない。
元科学評価の比較表登録も0件。今回の履歴は移植検収用の新attemptとしてのみ保持する。

## 実施した開始と同じrunの再開

925の通常`run_rpe_schedule_search.py:main(driver_factory=...)`から、既存make_executor/共通queue/PBS/worker/ingest/adoptを使う。
元結果を先に読むChainBackendを使わず、元の完全REQUESTを既存JobStoreへ1件だけ予約する外部technical callerを用意した。
通常Windows/Gateway Startにはexact保存REQUESTの強制再評価指定がないため、この一件はそのCLIからの実H4起動検収ではない。
人工controller hook・科学kernelのmock・科学SOURCEの編集は使っていない。

RomuTailで実施したcommandは次のとおり。同じ既存outputなら完了済みattemptを再利用し、追加Q3・PBS割当は作らない。
新しいoutputを作って再計算する操作は今回の一件許可には含まれない。

```bash
work='/home/abe/Project/Partially Randomized Trotter/.worktrees/rpe-pbs-h4-q3-portability-20261118'
base='/home/abe/Project/Partially Randomized Trotter'
"$base/.venv311/bin/python" -B "$work/scripts/check_rpe_pbs_saved_q3.py" \
  --service "$work/docs/research/rpe_pbs_h4_q3_portability_20261118/deployment/SERVICE.json" \
  --request "$work/docs/research/rpe_pbs_h4_q3_portability_20261118/selection/REQUEST.json" \
  --output "$base/.rpe-central/state/runs/pbs-h4-q3-portability-20261118/output" \
  --fixed-code-root "$base/.rpe-central/releases/925985b4e"
```

初回controllerは独立sessionで起動し、stdinを切り、run親directoryの`CONTROLLER.log`と`LAUNCH.json`へ記録した。
チャットに依存せずPBS投入・回収を続け、完了条件1件で停止した。
現在は完了しており、このcommandの返す`scientific_evaluation_calls=1`は保存された累計。再開差分は0である。

## 状態・停止・復帰

状態はrunの`PORTABILITY_STATE.json`、`PORTABILITY_RESULT.json`、`distributed/QUEUE.sqlite3`を読む。
今回の終了後状態は`distributed/pbs/FINAL_SNAPSHOT.json`に保存した。
`distributed/pbs/STATE.json`のRは終了前の最後のpollであり、FINAL_SNAPSHOTとPBS履歴を参照する。

```bash
cat "$base/.rpe-central/state/runs/pbs-h4-q3-portability-20261118/output/PORTABILITY_STATE.json"
ssh prism /opt/pbs/bin/qstat -xf -F json 267197.CPUcluster-PRISM
```

稼働中の通常executorへ新規受付停止を要求する既存markerは`output/STOP_REQUESTED.json`。
受領済みtaskは既存drain/UNKNOWN/retry契約を保持する。今回は完了済みで、停止操作は不要。
科学STOP/UNKNOWNや不明なstarting markerがある場合、別output・別candidateを作ってやり直さない。

PBS待機・保留・切断では同じrunとtokenを照合する。
今回のtokenは`137e68c230b348359c080cee211791cc`。
計算後の転送障害は元attemptの`collect`/`harvest`で回収し、追加の実Q3を起動しない。
完了後harvestの原36ファイルは[回収照合](results/AFTER_EXIT_HARVEST.json)ですべて一致した。
今回の再開ではqueue events・予算・採用・OUTCOME hash・PBS tokenが不変だった。

## 接続・配置・資源

SSH aliasはRomuTail既存の`prism`（akeno経由）。CPUcluster-PRISMをSSH aliasと仮定していない。
専用rootは`/home/AbeHiromu/projects/rpe-pbs-20261118`。

| 配置 | 相対path |
|---|---|
| 固定transport/gateway | `releases/925985b4e` |
| 専用Python3.11.0rc1 | `runtime/venv311rc1/bin/python` |
| 科学/route/adapter view | `readiness/views/<full SHA>` |
| 保存入力Git object | `readiness/objects.git`、不足9 object/71,096 bytesだけ補充 |
| 今回の事前検査・cache | `portability-h4-q3-20261118` |
| 今回の共有spool | `runs/a00784c07c478d880880` |
| taskの永続成果 | 上記spoolの`workers/pbs-0/98693b829295fc884330/18c2c7bc2d7b4a5581ef2543b1359e1b` |

計算nodeから中央への直接通信は使わず、共有homeをgatewayが配送した。必要成果をnode-local TMPDIRだけへ置いていない。
PBS stdout返送を待たずtask単位で回収・採用した。

workq、1node/8CPU/8GB/15分、DP8、library threads1、同時Q3数1、task memory6GiB、task_seconds.q3=480、drain60秒。
全96 CPUやworkq既定350GBを使った要求ではない。今回の保存資源実績に合わせた値である。
実割当を読み、枠上限1でworkerを動かした。待ちtaskがなくなると割当を解放し、qdelによる強制終了は不要だった。
controller終了後のoptional転送metric/最終pollにはOFFLINE記録があるが、元taskはTRANSFERRED・結果採用済みで、workerは終了した。

## 中央担当へ渡す設定と不足

このbranchのcaller/tests/選定・検収証拠を取り込み可能。通常の科学エンジン925自体は変更していない。
固定925の検収を、後続の異なるSOURCEの検収へ付け替えない。
共有currentの切替、staging Gateway・既存relayの所有整理は中央統合担当が行う。

終了後harvestで、system Pythonの整数JSON読込み制限（7168桁>4300）が見つかった。
今回の回収確認は`gateway_python`だけを既存専用Pythonへ向けた読み取りclientで復旧した。
固定runのSETTINGSは変更していない。[復旧設定](deployment/COLLECT_RUNTIME_OVERRIDE.json)。
**新規のPBS設定ではgatewayのJSON読込みruntimeもこの専用Pythonへ明示することを中央担当へ推奨する。**
実科学workerを管理サーバーへ移す変更ではない。既存runtimeでのファイル配送だけを担当する。
後続版でsystem Pythonを使い続けるなら、巨大なexact整数を保持するJSON読込みの技術修正・検査が必要。

今回のreal検収はFullPFの固定M5一件のみ。他方式実Q3・実信号・実回路compile・価格fitは未確認。
full探索の通常CLIに一件再評価モードを接続する場合は、今回の元REQUEST/科学identity保持とbudget1の経路を再利用する。
新しい科学scopeや追加の実Q3検収は、利用者の別途採択が必要。
追加の資源性能測定や科学結果表の重複登録は行わない。

次の担当は中央統合担当：公開commitと回収runtime設定を取り込み、共有currentの扱いを決定する。
この一件の再計算や、PBS残存jobの清掃は必要ない。
