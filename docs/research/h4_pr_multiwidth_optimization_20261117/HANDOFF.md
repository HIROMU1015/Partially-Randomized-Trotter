# H4 PR複数幅実探索 R06：稼働確認・同run再開・結果回収

現在の状態・PID・tmuxは [STATUS.json](STATUS.json)。採用統合版は **5829a7288d28395d2b3e89bfb676c23c690b9ddb**、限定実行修正は **7f1e267537f8a5aed76ea1447d7a5e82a678c314**。[SOURCE](SOURCE.json)に入力hashと元版を固定した。実行許可は今回の利用者指示で得ており、候補別の再承認やgrant発行は不要。

## まず動作中か確認する

host `GPUA100-gleap02`。worktree:
`/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-pr-multiwidth-optimize-20261117`

branch `codex/b2-rpe-pr-multiwidth-optimize-20261117`、round `H4_PR_BOTH_MULTIWIDTH_OPTIMIZATION_R06_20261117`。
run directory:
`docs/research/h4_pr_multiwidth_optimization_20261117/run01`

```bash
tmux has-session -t h4-pr-width-r06
tmux list-panes -t h4-pr-width-r06 -F '#{pane_pid} #{pane_dead} #{pane_current_command}'
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-pr-multiwidth-optimize-20261117
tail -n 12 docs/research/h4_pr_multiwidth_optimization_20261117/run01/CONTROLLER_INVOCATIONS.jsonl
tail -n 12 docs/research/h4_pr_multiwidth_optimization_20261117/run01/DRIVER_PROGRESS.jsonl
tail -n 12 docs/research/h4_pr_multiwidth_optimization_20261117/run01/PROGRESS.jsonl
```

sessionなしだけで科学worker完了とは判定しない。STARTのpid、実際のcmdline、所有childのPROCESS_RETURN/JOB_STAGESを確認する。CPU生存だけで前進・gate PASSとも判定しない。実時間と段階の終了をログで見る。faulthandler定期dumpは使わない。

## 正確な開始・再開コマンド

まず上でcontroller/workerの生存を確認する。起動と同run再開は同じwrapperを使う。

```bash
bash /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-pr-multiwidth-optimize-20261117/docs/research/h4_pr_multiwidth_optimization_20261117/START_OR_RESUME.sh
```

wrapperは所有runのflockを保持し、二重起動をexit75で拒否する。元の通常CLIを一度起動し、科学retry loopを持たない。tmux起動では次を使う（同名sessionがないと確認した場合のみ）。

```bash
tmux new-session -d -s h4-pr-width-r06 'bash /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-pr-multiwidth-optimize-20261117/docs/research/h4_pr_multiwidth_optimization_20261117/START_OR_RESUME.sh >> /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-pr-multiwidth-optimize-20261117/docs/research/h4_pr_multiwidth_optimization_20261117/run01/CONTROLLER.log 2>&1'
```

実際にwrapperが呼ぶ入口は次の通り。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python -B \
scripts/run_rpe_schedule_search.py \
 --problem docs/research/h4_pr_multiwidth_optimization_20261117/PROBLEM.json \
 --round docs/research/h4_pr_multiwidth_optimization_20261117/ROUND.json \
 --run-dir docs/research/h4_pr_multiwidth_optimization_20261117/run01 \
 --batch 64 --max-cycles 200
```

`--plan-only`を本番へ付けない。準備時のROUND_PREPARATION/preflight_runは科学0の別検査で、部分markerを本番へ転用しない。正常な科学処理をchat終了のために停止しない。継続不能な異常、UNKNOWN、旧one-shot条件は既存SOURCEとSTOPに従い、collect可能な完了workerを新規実行に置換しない。SOURCE・入力・round条件を動作中に書き換えない。

## 実行scope・再利用・性能

PR-Pauli=[数値0.1,数値0.2]、PR-DF=[数値0.1,数値0.05]、各LD1〜9。Pauli0.2はbinary64のexact値で、文字列1/5へ変換しない。κは既存mapperの異なるscheduleごと、shotは既存V2。h/T/target/decoder/law/loss/exactτを保持し、各候補自身の信号・26軸価格を使う。元compiler/lowering/測定付きwrapperを変更しない。

開始点：Pauli410199237 RZ、DF155652932221/126 RZ。[STARTING_EVIDENCE](STARTING_EVIDENCE.json)にC fee61・全完了4783件の固定locator、各点のκ/配分/MSE/元receiptを保存。round03/04を含むprior pathは元所有worktreeへの絶対path。同じSOURCE/軸/law/実幅/契約の完了を再利用し、未解決なjobは二重投入しない。旧round05と失敗start/STOPを保全する。

設定はconcurrent16slots・DP3workers・prefix-cache m5/m11・HYBRID・overlap。準備は元per_decision8/方法/cycleに区切り、全222初期pointを一括処理しない。未選択は次cycleへ保持し、既知の安いfamilyと未試行幅へ機会を配る。source・scopeの修正と36人工回帰はREPORT/launch_contracts参照。実H4の速度倍率は未測定。

確認先：`KAPPA_STATE.json`（scope/選択/保留）、`completion/STATE.json`・`completion/PROGRESS.jsonl`（科学stage）、`STEP_PRICE_RULES.json`（実幅別model、実際の場所はCompletion内）、`ROUND_RESULT.json`、`CONFIRMED_MINIMA.json`、`INVOCATIONS.jsonl`、job別`DP_PREFIX_EVENTS.jsonl`と元API/PROCESS_RETURN。DP途中状態を使った別配分Q3は新規Q31件、完全一致job再利用は0件で区別する。

## 終了後の集計・C同期

方式別に①開始時②0.1探索後③新幅④全幅の最安を並べる。`rule_adopted_best`と`confirmed_all_best`は別欄。小さい改善でもidentity・自身の単価・exactτを通過した完了Q3は確認済み最安に含める。各点のLD、要求/実幅、絶対κ、q/R、N_X/N_Y、MSE上下界、条件付きC_total、開始時との差/率と元receiptを残す。direct/model/外挿、未評価/入力不足/UNKNOWN/配分不適合を分ける。全候補最適・正式G/winnerを認定しない。

通常C同期は既存`codex/b2-rpe-*`発見条件を利用する。実際の結果は既存`publish_comparison_exports`／`export_pr_candidate_comparison.py`で各候補の実信号/価格/元Q3をC_RESULT_JOINへ結び、固定commit/pathを公開する。必要なら新幅contextを同Completion.installで復元して元Executor.prior_recordsに渡す。古いround_contextだけで未登録の新幅を消さない。未採用完了も検査し、単に採用traceだけから最安を決めない。C branch/cronには書き込まず、元consumerの読取確認を残す。

実行中のchat終了では最終集計を済ませたとは扱わない。次担当はこの同runを回収し、必要な後処理・REPORT更新・commit/push・remote照合を行う。ロードマップ本体は統合担当へ渡す。

## 起動済み・復旧境界

起動時SOURCE b0257d78026ade89bb3ba61dc88634570039dae8、controller PID32111、wrapper PID32104、tmux h4-pr-width-r06。[LIVE_START_SNAPSHOT](LIVE_START_SNAPSHOT.json)に実requestのprofile・同時稼働・HIT・全起動と個別結果・保留を記録。snapshotは時点情報であり、今動いているかを先に確かめる。

Pauli初期8件は新信号0件の旧STOPとして残す。すでに固定されたA_SOURCEに含まれる不足module2件だけを配置し、caller/環境/ROUND/数値規則を変更していない。配置SOURCE/blobとguard証拠はDECLARED_DEPENDENCY_DEPLOYMENTを参照。新たな数値SOURCEをライブで選び直したり、旧8件を自動retryしたりしない。legacy descriptorの形式不一致とDF4件の保存template不足はclean SOURCE/接続での将来対応が残る。4件とも既存科学結果の回収ができると仮定しない（API前失敗）。現在の正常較正をchat終了のために中断しない。
