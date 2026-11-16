# B2への引渡し：4方式共通RPE scheduler

実装SOURCE：`c3bb2ad2c409c48ab73708ca61aeab5b9323b317`。専用branch：`codex/d2-rpe-optimizer-throughput-20261116`。基点の実行SOURCE `1e88cd85cc8d2739a4dce43d138b94b153309333`、公開 `98b6f8e1ac6e68285d2c68a8e35125d02d6edd9c` を保全した。結果の公開commitは完了報告のremote SHAを使用し、このディレクトリの各pathを参照する。

共通queueを使うopt-in高速版を実装した。人工4方式では元と同じ最安点・採否配分順・停止理由で、遅延workloadの中央値2.537→0.722秒、対比較の速度比中央値3.52倍。単一方式9人工構成の18-slot飽和ケースは1.043→1.179秒で短縮しなかった。**実Q3の速度改善は未確認**。新規実Q3・信号・path・build/compile・fitは0件。

現行B2の完了済み3,353 jobのログでは平均同時17.992/18、利用率99.955%、起動→科学call平均5.674秒、DP平均117.601秒。先読みの実採用率と候補待ち帰属時間は旧logからは分からず、nullを維持した。今回B2のprocess、実行directory、checkpoint、SOURCEには書き込んでいない。

## 読むもの・適用箇所

|locator（commitは上記SOURCE）|役割|
|---|---|
|`src/trottertracks/rpe_optimizer_concurrent.py`|共有queue、予約、起動/採用計測、再開|
|`scripts/run_rpe_method_optimizer.py`|元のround CLIにscheduler選択を接続|
|`scripts/run_rpe_optimizer_q3_job.py`|元のroute adapter。既存DP workerパラメータだけを選択可能にした|
|`tests/test_rpe_optimizer_concurrent.py`|通常・有限予算・重複・逆順完了・再開・CLIの人工15検査|
|`scripts/benchmark_rpe_scheduler.py`|同じ開始状態の基準/新版人工比較、3反復|
|`scripts/check_rpe_scheduler_saved_replay.py`|固定C catalog再生、実Q3 0|
|`scripts/check_rpe_scheduler_constructor.py`|固定Pauli common routeの元AST constructorのみ、kernel呼出し0|
|`scripts/analyze_rpe_scheduler_logs.py`|完了jobの運用logだけを読む。科学RESULT/OUTCOMEを開かない|

公開結果は同ディレクトリの [REPORT](REPORT.md)、[SOURCE](SOURCE.json)、[OBSERVED_LOGS](OBSERVED_LOGS.json)、[BENCHMARKS](BENCHMARKS.json)、[集約](BENCHMARK_SUMMARY.json)、[保存再生](SAVED_REPLAY.json)、[constructor](CONSTRUCTOR_CHECK.json)、[検査log](TESTS.txt)。基準/新版のexact人工最安点、採否vector順、各構成STOPはBENCHMARKSの各row.searchに保存した。保存再生の価格や科学MSEは元入力を使用し、再評価していない。

## 次の承認済み探索で使う設定・起動command

次の探索用の**新しいrun directory**で使用する。実行中κ探索に設定を差し込まず、完了済みrunを高速化のため再実行しない。探索のmethod、候補範囲、LD/κ、初期状態、budget、decoder/target/loss/τ、routes、価格入力は次roundの採択内容をそのまま使う。今回の技術実装から新たな科学実行許可は作らない。

新round JSONの `executor` に次を設定する（機械可読fragment：[EXECUTOR_PROFILE.json](EXECUTOR_PROFILE.json)）。

```json
{
  "scheduler": "concurrent",
  "max_concurrent_jobs": 18,
  "dp_workers": 8,
  "wait_for_running_jobs": true
}
```

18/8は既存運用の初期値を保った設定で、実Q3の最適profileを認定した値ではない。slot/workerを変える場合はこの2値を明示し、同時数・RSS・DP所要時間を読む。lookaheadは元roundの値を継承する。次runが単一方式で常時飽和する形なら、旧schedulerの維持も選択肢であり、新版が速いとの測定根拠は今回ない。

次の承認済みroundを準備した後の起動例（round fileのbudget等は変更しない）：

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/d2-rpe-optimizer-throughput-20261116
# 次の採択済み範囲・保存SOURCE・保存結果を指定したround JSONを入力する。
NEXT_TASK_ROUND=/absolute/path/to/next_adopted_round.concurrent.json
NEXT_TASK_RUN=/absolute/path/to/new_next_search_run
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_rpe_method_optimizer.py \
  --problem configs/rpe_method_optimizer/h4_df10_problem.json \
  --round "$NEXT_TASK_ROUND" --run-dir "$NEXT_TASK_RUN"
```

controllerは今回Python 3.12、実Q3 childは各routeに固定されたinterpreterを使う。固定Pauli common routeはPython 3.11.0rc1で元AST guardを通した。Full PF/PR-DFを含む4方式のscheduler入口は人工subprocessで検証し、各実routeの科学評価は今回していない。

完了後の別roundでは、前roundの `ROUND_RESULT.json` をroundの `resume_from` またはCLIの `--resume-from` で引き継ぎ、既存catalog/公開結果も従来どおり参照する。source不明の結果は採用しない。job単位の途中再開は同じround JSON・run directory・commandで行う。保存OUTCOME、起動済みjob、未採用の先読みOUTCOMEも同じjob identityで再利用する。同directoryにlegacyとconcurrent driverを同時起動しない。

## 探索規則・予算・再開の意味

元の `proposals`、`method_state`、`run_method`、`run_by_configuration` は固定SOURCEとAST一致。構成内の採否順は元の規則を保ち、先に返った別構成や先読みの回答で順番を変えない。方式間は独立に進め、空いたglobal slotを共有する。METHOD scopeにも同じ非同期prefetch入口を接続した。

有限budget時は従来どおり構成を順に扱い、通常探索は追加の先読みを行わない。queueへの予約時に既存起動と合わせて上限を検査し、先読みを含む起動済み件数をbudgetとして計上する。未起動の古いqueueだけを取り消せる。起動済みworkerを先読み破棄のために停止しない。

再開したlive workerもglobal slotに数える。slot設定を減らしても既存workerを止めず、その分だけ新規開始を待つ。完了結果は元のidentity/contract検査で受け入れる。STARTINGのみで起動状態が不明なら無断retryせず要確認。Popen後のreceipt保存失敗でもslotを保持し、起動済みchildを終わらせて追加投入を止める。`executor_failure` に理由を残す。dead launchは元のcollect（科学計算なし）で結果保存物を確認し、回収不能なら従来のSTALE/UNKNOWNとして扱う。

集計では次を混同しない。

- methodの `new_calls`：今回実際に起動したjob数（未採用先読みを含む）。`adopted_new_calls` と構成別trace.callsは従来の採用時計数。
- `executor_metrics.launched/adopted/speculative_launched/speculative_adopted`：今回invocationの運用計数。`new_real_route_launches` は実routeへの起動数で、科学kernelの成功回数を意味しない。
- `completed_not_adopted`：このinvocationで起動し、採否順に使われなかった結果。rawは保持する。
- `PROGRESS.jsonl`：ENQUEUE/LAUNCH/ADOPT/EXIT/RESUME/CANCELをepoch・monotonic・invocation付きで保存。起動までのqueue_wait_seconds、終了code/errorを保存する。
- slotの空き時間はqueue有無で分ける。内蔵RSSは所有するjob treeの100ms標本、shared pageの重複を含み得る。CPU使用や生存を科学PASSの代用にしない。

## 検証の再現

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 tests/test_rpe_optimizer_concurrent.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/benchmark_rpe_scheduler.py \
  --output /tmp/new_unique_d2_scheduler_benchmark --repetitions 3
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/check_rpe_scheduler_saved_replay.py \
  --output /tmp/new_unique_d2_saved_replay.json
PYTHONDONTWRITEBYTECODE=1 /home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python \
  scripts/check_rpe_scheduler_constructor.py \
  --view /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/b2-route-view-b-bd42dd1e9 \
  --output /tmp/new_unique_d2_constructor_check.json
```

これは保存読取・人工worker・constructorのみの技術検証。H4/P0回帰suite、実kernel評価、同じκ探索の複製は起動しない。報告値は固定SOURCEでの3反復。実ホストを隔離しておらず、実DP最適worker数・厳密最大メモリ・科学性能は未確認。

## ロードマップ・終了

共通探索基盤にscheduler選択と運用計測が追加された。既存価格資格、科学Q3、最安値、LD/κの判断は変更しない。次の承認済み多方式探索で、新しいlogから採用率・queue待ち・起動とDP時間・メモリを比較できる。現行B2探索はそのまま継続でき、終了を待つ必要のある改修はしていない。

原SOURCE・入力・raw・旧STOPを保全した。今回の公開とremote照合で終了し、次の科学探索は開始しない。
