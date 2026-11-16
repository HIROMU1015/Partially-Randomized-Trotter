# 固定28d：plan記録と次の実H4 run案

今回実行したのはplan-onlyと明示dry-checkのみ。以下の科学run用コマンドは未実行である。
実2台・3台人工検収、fault検収、42件のテストは再実行していない。

## 今回使用した固定入力と通常入口

SOURCEは `28d976b99cdf8329adcc1b79327e329e947ef774`。
専用viewは `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-q3-source-28d976b99`。
計画branchは `codex/rpe-distributed-h4-plan-20261117`。problemとroundはこのbranchの
`docs/research/rpe_distributed_h4_plan_20261117/inputs/` にある。

今回の通常入口の最終呼出し（実行済み、終了code0、新Q3 0件）：

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-q3-source-28d976b99
/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python -B \
 scripts/run_rpe_schedule_search.py \
 --problem /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-h4-plan-20261117/docs/research/rpe_distributed_h4_plan_20261117/inputs/PROBLEM.json \
 --round /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-h4-plan-20261117/docs/research/rpe_distributed_h4_plan_20261117/inputs/FULL_HISTORY_PLAN_ROUND.json \
 --run-dir docs/research/rpe_distributed_h4_plan_20261117/local_plan/full_history --plan-only
```

再現する場合は既存結果を保持し、`--run-dir` を新しい計画用directoryへ変える。
このroundは全方式budget0・science=false・legacy read-only prior利用であり、分散queueを作らない。
方法固有の候補・26軸価格は既存mapper／PriceRulesを使う。担当者がLDやpを手動指定していない。
初回catalog-only候補の重複が分かったため、最終版は旧PF全履歴の2runもpriorに含めた。

PRの幅・κの補助計画は `checks/WIDTH_FRONTIER_RUNNER.py` が固定W helperを呼ぶ。
これは通常入口の実行結果とは別の補助metadataで、R06の既存239点・準備要求と照合する。
固定28dの混在方式計画と、R06採用済みの方式別計画の差を記録しており、SOURCEへ変更を加えていない。

## サーバー2の実入力dry-check

4候補の `assignments/*.json` は配送依頼ではなく、workerの入力検査を直接使うための
READINESS_ONLYデータである。科学job IDは既存Executor.job_dirと同じkeyから組み立てた。
attempt欄は `READINESS_ONLY_NO_EXECUTION_...` として実行attemptと区別した。
各hostの `route_views`・interpreter・DP/cache pathだけをprofileで解決する。

今回使用したコマンド（`METHOD` はFULL_PF/TRUNCATED_PF/PR_PAULI/PR_DF）：

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-h4-plan-20261117
/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python -B \
 scripts/check_rpe_q3_host_readiness.py \
 --assignment docs/research/rpe_distributed_h4_plan_20261117/assignments/METHOD.json \
 --profile docs/research/rpe_distributed_h4_plan_20261117/profiles/server2.json \
 --output docs/research/rpe_distributed_h4_plan_20261117/dry-check/server2/METHOD
```

再検査は新outputへ保存する。上記toolはWorkerを起動せず、固定workerの
`validate_assignment` とadapterの明示 `--mode dry-check` だけを使う。
SOURCE8ファイル、入力blob hash、route HEAD、runtime、元の評価契約・count scopeを検査する。
科学Q3、collect、mailbox status/submit、HTTP controllerは呼ばない。
遠隔検査は [LOCAL0_DRY_CHECK](LOCAL0_DRY_CHECK.md) の一箇所へまとめた。

## 次回の科学実行scopeと設定案

[NEXT_RUN_ROUND_DRAFT](inputs/NEXT_RUN_ROUND_DRAFT.json) は**science=falseの未実行案**。
初回は既存規則が選んだFull PF・B0I・PR-DFの独立Q3を各1件、計最大3件だけとする案である。
信号生成・価格較正・回路compile・新L_D/κ/幅/pの科学準備は含めない。
PR-Pauliは実入力gate不適合と最新最安の軸価格binding不足のため、案の実行方式から外した。
3枠はこの3つの独立要求を同時に扱うための設定例で、資源の恒常的上限や最適profileではない。
Full PF/B0Iの最初の配分はglobal最安より高いEXPLOREであり、初回PASSだけで最安更新するのはPR-DFのみ。

現在実機PASSはサーバー2だけなので、この案の参加hostはserver2だけ。
同じ分散executorを使い、server2が唯一のcontrollerかつ計算workerになる。
価格・route registryは今回の固定snapshotを参照する。既存prior9runを全て参照し、同じ科学job keyを再投入しない。
R06のUNKNOWN4件とB0IのUNKNOWN1件、過去のSTOPを解除しない。
実行直前までに新しい保存結果・所有状態が変わった場合は、その候補だけ重複照合を更新する。

[THREE_HOST_ROUND_DRAFT_NOT_READY](inputs/THREE_HOST_ROUND_DRAFT_NOT_READY.json) は
同じ候補・予算でserver1/localを参加させる未準備案。
各候補×全参加hostの実機dry-check PASSが先に必要である。
固定schedulerにはroute能力別のhostフィルタがないため、PR-DF非対応hostを登録したまま
queueへ仕事を入れる案にしない。方式・LDをhostへ固定して回避もしない。
前回の人工profile/spool/relayを本番へ流用せず、新run用のprofile・state・spoolを用意する。
local0の接続方向はqg2/qg1/RomuTailの既存aliasを使う前回方式を維持する。

科学実行範囲の利用者判断後、採用する新しいroundを固定し、既存のscienceフラグを
その範囲に合わせる。承認証明用のファイルやgrant発行は必要ない。
以下の `ROUND_SCIENCE.json` は採用した新round設定の配置先を表す。現在は作成していない。

## 将来の開始・再開・状態確認

サーバー2での開始と同run再開は同じ通常入口を使う。
現在の計画directoryや過去科学runを `--run-dir` に指定しない。
SOURCE・problem・round・予算・host ID/spoolを同run途中で変えない。

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-q3-source-28d976b99
/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python -B \
 scripts/run_rpe_schedule_search.py \
 --problem /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-h4-plan-20261117/docs/research/rpe_distributed_h4_plan_20261117/inputs/PROBLEM.json \
 --round /home/AbeHiromu/projects/rpe-distributed-h4-next-run-20261117/ROUND_SCIENCE.json \
 --run-dir /home/AbeHiromu/projects/rpe-distributed-h4-next-run-20261117/run01
```

同runの二重controllerを避ける既存flockと、永続割当を維持する。
状態確認は `run01/PROGRESS.jsonl`、`EXECUTOR_METRICS.json`、方式別ROUND_RESULT、
`distributed/QUEUE.sqlite3` とjobのPROCESS_RETURN/JOB_STAGESを読む。
local0のrelay停止は新run用configへの `run_rpe_q3_relay.py --config ... --stop`、再接続は
同じconfig/stateで `--start`。受信済みQ3は停止させず、同job/attemptで回収する。
科学controllerの停止は既存のrun所有者の手順・固定契約に従い、通信断や他chatを理由に
workerをkill/retryしない。今回これらの起動・停止操作は行っていない。

## 次runで残す指標

- 方式別の開始時・各完了時の最安費用、構成、X/Y、exact MSE上下界、採用/未採用完了を別記。
- 実Q3完了件数/時：新科学評価だけを分子にし、完全一致job再利用を別件数にする。
- controller全wall、最初の科学LAUNCH〜最後のEXIT、入力準備・結果回収を別区間にする。
- host別起動/完了/採用/再利用件数、queue予約〜割当のcount/mean/max/総和。
- PROCESS_RETURNの計算時間、controller検証・永続化時間、SSH/HTTP upload〜ACK時間を区別。
  欠測を0で補完せず、並行待ち総和をwallや速度比にしない。
- DP_PREFIX_EVENTSのHIT/MISS・donor・再利用round数、host別cache、DP worker数、終了code/signal。

比較用の保存単機locatorは `checks/PF_INVENTORY.json.gz` の各job/PROCESS_RETURN、
`checks/SERVER2_INVENTORY.json` のR06各jobと元INVOCATIONS。
新しい候補のshot配分が異なるので、そのまま同一負荷の速度比にはしない。
固定SOURCE/route・実入力・DP profile・cache温度・候補集合の対応が確認できる部分だけで比較する。
今回、実H4の分散速度倍率を測定・推定していない。
