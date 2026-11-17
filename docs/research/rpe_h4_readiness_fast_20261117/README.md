# H4：候補・実行可否を1コマンドで返す

通常入口に `--plan-only --prepare-readiness` を付ける。
新しい結果directoryを指定し、2回目以降は同じ `--preparation-cache` を使う。
前回33928bfのproblem/round、既存選択規則、固定計算SOURCE28dを使用する。

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-h4-readiness-fast-20261117
/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python -B \
 scripts/run_rpe_schedule_search.py \
 --problem docs/research/rpe_distributed_h4_plan_20261117/inputs/PROBLEM.json \
 --round docs/research/rpe_distributed_h4_plan_20261117/inputs/FULL_HISTORY_PLAN_ROUND.json \
 --run-dir /home/AbeHiromu/projects/rpe-h4-preparation/NEW_RESULT \
 --plan-only --prepare-readiness \
 --preparation-cache /home/AbeHiromu/projects/rpe-h4-preparation/cache
```

`--prepare-readiness` は `--plan-only` が必須。全新Q3予算を0にし、
候補の科学的取得や本番queueを起動しない。実行済みのrunや前回の遠隔出力先を指定しない。
対象は前回の混合方式method-round frontierであり、既存の幅plannerや科学実行の既定動作は変えない。

工程は、固定入力確認→metadata context再利用→適合・不適合を含む履歴の差分照合→
既存optimizerで未評価配分の選定→REQUEST組立→server2の並列dry-check→READY。
UNKNOWN・起動済み・未回収のmarkerとprocess、既存分散queueがあればその状態は毎回読み取る。
同jobの既所有や状態不明を実行可能と認定しない。

`CANDIDATES_AND_REMOTE_INPUTS_AVAILABLE` のREADY行が出たら判断・遠隔検査に進める。

- `READY.json`：候補、現在最安との差、乾式判定、元結果への参照、現在の未解決状態。
- `readiness/requests/`・`assignments/`：既存job keyで組み立てた乾式検査用REQUEST。
- `readiness/LOCAL0_DRY_CHECK.zip`：固定入力が同じなら前回ZIPとbyte一致。
- `plan/ROUND_RESULT.json`：小さいplan要約。exact原結果のresumeファイルとしては使わない。
- `TIMING.json`・`PREPARATION_STAGES.jsonl`：READYまでとREPORT終了までの別時計。

今回の4REQUEST・手順・旧remote出力先は、引き続き
[固定33928bfのLOCAL0_DRY_CHECK](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33928bf80cca18ce3f1fa3ddde88d27d7e2b7e5b/docs/research/rpe_distributed_h4_plan_20261117/LOCAL0_DRY_CHECK.md)
を使用する。旧入力や手順を新しいpathへ置き換えない。
新着結果で候補が変わる場合は別名の新bundleを出し、既存の遠隔作業と混ぜない。

cacheはこの利用者がこの手順で生成したlocal SQLite／圧縮pickleだけを使う。
exact値を原OUTCOME・出所hashとともに保持し、他hostから受け取ったpickleを読み込まない。
削除しても元結果を失わず、初回に原結果から再構築する。
contextは固定入力・metadata実装・provider・価格registry等が変われば再生成し、
履歴は新着・変更分を既存job key／route／identityの検査へ戻す。
乾式判定の再利用は現在のSOURCE・入力・route・runtimeと元保存結果hashが一致する場合だけ。
cached failureはfailureのままで、新しい実機PASSとは数えない。

再計測は、まだ存在しないbase directoryを指定する。

```bash
python3 -B docs/research/rpe_h4_readiness_fast_20261117/RUN_BENCHMARK.py \
 --output-base /home/AbeHiromu/projects/rpe-h4-readiness-benchmark-NEW
```

親processが通常コマンド開始前からREADY受信、REPORT完了、process終了を測る。
GitでのREPORT公開完了は [PUBLICATION_TIMING](PUBLICATION_TIMING.json) で別に報告する。
[REPORT](REPORT.md)／[実測](METRICS.json)／[同値検査](EQUIVALENCE.json)。
