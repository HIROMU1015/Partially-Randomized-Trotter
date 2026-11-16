# B2への引継ぎ：H4 RPE探索の共通queue・DP途中状態・準備待ちの利用

2026-11-17 JST。正本は [REPORT](REPORT.md)、固定版は [SOURCE.json](SOURCE.json)。branchは `codex/h4-rpe-search-throughput-20261116`、基点はB2 `d8e806265c3ded8e1184181c37c2d7ec71f08ade`。B2のworktree、原run、固定SOURCE、旧STOPは変更していない。今回の実H4信号・較正・compile・Q3・round05本番は0件。

## 取り込むもの

通常入口 `scripts/run_rpe_schedule_search.py` → `scripts/run_rpe_kappa_neighbourhood.py` → `scripts/run_rpe_method_optimizer.py` のまま使う。D2 helperを利用し、B2のPR-DF共通入口・旧A入口互換・process gate・collectを保持した。

- `src/trottertracks/rpe_optimizer_concurrent.py`：共通queue、予約・真の起動予算、prior再利用、live再開と終了待ち、時刻・待ち理由。
- `src/trottertracks/rpe_dp_prefix_reuse.py`、`scripts/run_rpe_optimizer_q3_job.py`：固定AST guardを保った途中状態とworker profile。
- CLIと `pr_candidate_completion.py`：準備後の継続にも同じoptimizerを渡す。新しい準備/科学workerループは作らない。
- `rpe_priority_search.py`：選択式の適応HYBRIDとshot振替。既定は無効。
- `scripts/prepare_rpe_throughput_round.py`：原roundを書き換えず、新run・新設定例・route/価格登録表を作る。既存run/outputがあれば上書きせず終了する。

検証済み実装commitは `a50aadc1f430e8c8ee6f8917a8024f1121ec913a`。必要なコードは一式を取り込む。古いD2 CLI全体でB2を置き換えると、B2後続のroute/collectを落とすため避ける。実装commitと資料commitの区別はSOURCE.jsonに記載する。

## 最小設定

次の設定を、次に採択されたroundのexecutorへ追加する。32slot/8workersはround05の既存運用値を維持した例で、最適値ではない。profileを変える場合も構成・信号・評価契約・予算はそのroundのものを維持する。

```json
{
  "executor": {
    "scheduler": "concurrent",
    "max_concurrent_jobs": 32,
    "dp_workers": 8,
    "wait_for_running_jobs": true,
    "prefix_cache_directory": "/absolute/path/to/new_owned_run/prefix_cache",
    "prefix_checkpoint_rounds": [5, 11]
  },
  "overlap_ready_batch": false,
  "search_overrides": {
    "lookahead_mode": "HYBRID",
    "adaptive_hybrid": false,
    "allocation_exchange": null
  }
}
```

既存の `prior_run_dirs`・ledger・method/route/search設定は保持する。相対 `prefix_cache_directory` は新runを基準に解決する。checkpointに保存するのは科学状態であり、承認記録ではない。原OUTCOME/rawと分けて所有するcacheを使う。旧OUTCOMEにはfrontierがないので、過去の結果だけからcacheを作らない。

## 設定例の作成と、科学計算を行わない入口確認

下の例は新しいpathを使う。`--saved-root` は保存結果を持つ元B2 checkoutの絶対path。generatorは既存の固定価格登録表もコピーするため、初回quote前に表が存在する。新設定の `science_approved` はfalseである。

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-rpe-search-throughput-20261116
RPE_SAVED_ROOT=/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/b2-rpe-split-preparation-20261116
RPE_NEXT_ROUND=/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-rpe-search-throughput-20261116/configs/rpe_method_optimizer/rounds/h4_pr_pauli_throughput_next.json
RPE_NEXT_RUN=/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-rpe-search-throughput-20261116/docs/research/h4_rpe_search_throughput_20261116/next_run
RPE_CONTROLLER_PYTHON=/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python

PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/prepare_rpe_throughput_round.py \
  --saved-root "$RPE_SAVED_ROOT" --output "$RPE_NEXT_ROUND" --run-dir "$RPE_NEXT_RUN"
PYTHONDONTWRITEBYTECODE=1 "$RPE_CONTROLLER_PYTHON" -B scripts/run_rpe_schedule_search.py \
  --problem configs/rpe_method_optimizer/h4_df10_problem.json \
  --round "$RPE_NEXT_ROUND" --run-dir "$RPE_NEXT_RUN" --plan-only
```

今回検証済みの設定は `entry/ROUND_VERIFIED.json`、公開する状態要約は `entry/ENTRY_SUMMARY.json`。重複metadataを含む `entry/verified_run/` 全体はサーバー上に保持する。検証結果は [PLAN_ONLY_ACCEPTANCE](entry/PLAN_ONLY_ACCEPTANCE.json)。開始・同run再開とも終了code0、LD1〜9のfamily constantsを維持し、Q3・信号・Completion実行0、較正request0だった。再開時も `--plan-only` を付ける。

## 次の科学実行が別途許可された場合の開始・再開

今回の技術作業は科学実行許可ではない。次の採択されたSOURCE・入力・条件・範囲の許可がある場合に限り、その新roundで `science_approved` と必要なscope設定を整える。signals/pauli workspace・branch・価格表・固定routeをB2自身が所有する経路へ照合し、稼働中のSOURCEや旧runへ設定を差し込まない。

開始・再開は同じコマンドである。κ・LD・shot数を人手で列挙する必要はない。

```bash
PYTHONDONTWRITEBYTECODE=1 "$RPE_CONTROLLER_PYTHON" -B scripts/run_rpe_schedule_search.py \
  --problem configs/rpe_method_optimizer/h4_df10_problem.json \
  --round "$RPE_NEXT_ROUND" --run-dir "$RPE_NEXT_RUN"
```

- 同じrunの完了job、未採用先読み、既存signal/価格/READY記録を従来のidentityで再利用する。prior jobの再利用はcalls=0。新しい配分のprefix利用は新Q3 1件。
- batchごとの予約も元roundの総予算を超えない。V2は方法全体の最新previewにない未起動予約を取消し、実行開始済みjobを停止しない。構成別探索は他構成のqueueを取り消さない。
- 以前のlive jobを枠に数える。slot数を減らしても既存jobを停止しない。`wait_for_running_jobs=true` なら未採用live jobも終了まで待つ。
- 適応先読みの観測窓は再開invocationでresetするため、未採用先読みの順序・件数が同一になるとは限らない。完了結果再利用・起動済み予算は失わない。
- `V2_UNKNOWN_RESULT`、stale、route不足、subprocess失敗等の保存済みSTOPを同runで自動的に開かない。計画専用で一度も実行しなかった準備要求だけを再開対象に戻す。
- fixed Pauli kernel/AST guardの検査は既存Python 3.11.0rc1 routeで行った。別runtimeへguard回避で移さない。controllerのPythonと科学childのinterpreterは別である。

## 任意の探索変更

`--adaptive`、`--overlap` をgeneratorへ渡すと、それぞれ `adaptive_hybrid`、`overlap_ready_batch` を有効にした新設定を作る。既存設定の再生成には使わず、採択された新roundで扱う。

```json
{
  "search_overrides": {
    "adaptive_hybrid": true,
    "allocation_exchange": {"enabled": true, "max_moves": 8, "add_steps": [1, 2]}
  },
  "overlap_ready_batch": true
}
```

適応先読みは独立候補を優先し、余り枠だけで連鎖を調整する。投入数減少が到達時間短縮になるとは限らない。振替は費用が下がる配分を生成するだけで、適合性は通常Q3で判定する。未知の保存配分を不適合としない。準備との並行では、既存READY候補が先に予算を使うため、全準備後の探索と結果が変わり得る。

## 記録・検証と未確認範囲

入口の `ROUND_STAGES.jsonl`、`DRIVER_PROGRESS.jsonl`、queueの `PROGRESS.jsonl`、jobの `DP_PREFIX_EVENTS.jsonl`・元DP stage・PROCESS_RETURN/子exit/signalを使う。CPU生存や利用率だけでPASSを判定しない。prefix HITの復元済みobservationsは以前のworkerの観測を含むので、今回計算したroundと区別する。

[検査一覧](checks/VALIDATION.json)、[保存分析](analysis/SAVED_OPERATIONAL_SUMMARY.json)、[人工DP33比較](prefix/SUMMARY.json)、[選択後開示の再生](adaptive/saved_replay/SAVED_REPLAY.json) を参照する。再現例は新しい出力directoryを指定する。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B tests/test_rpe_throughput_entry.py
PYTHONDONTWRITEBYTECODE=1 python3 -B tests/test_rpe_optimizer_concurrent.py
PYTHONDONTWRITEBYTECODE=1 /home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python \
  -B tests/test_rpe_dp_prefix_reuse.py
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/check_rpe_adaptive_throughput.py --output /tmp/new_rpe_adaptive_checks
```

実H4のwall/総CPUの短縮、同じ実予算での最安費用、実checkpoint容量、実較正との重なり、Full PF/B0I/PR-DFの効果は未測定。今回の保存PR-Pauli結果・小さい人工DPの倍率をそれらの認定に使わない。REPORTの優先順位は次の許可された実行で比較する技術的根拠であり、新しい科学計算を今回開始する指示ではない。
