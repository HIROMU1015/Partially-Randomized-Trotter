# R07：稼働確認・同run再開・結果回収

host `GPUA100-gleap02`、branch `codex/b2-rpe-pr-multiwidth-resume-20261117`。
worktree `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-pr-multiwidth-resume-20261117`。
科学code SOURCE `82dd08d6cf8b31545fd83d032cd64e1cfa5c09b5`（修正元44fadf8b8）。入力commitは起動時の`CONTROLLER_INVOCATIONS.jsonl`へ記録する。

round `H4_PR_BOTH_MULTIWIDTH_RESUME_R07_20261117`。
run directory `docs/research/h4_pr_multiwidth_resume_20261117/run01`、tmux `h4-pr-width-r07`。

まず生存と段階前進を確認する。別のcontrollerを重複起動しない。

```bash
tmux has-session -t h4-pr-width-r07
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-pr-multiwidth-resume-20261117
tail -n 8 docs/research/h4_pr_multiwidth_resume_20261117/run01/CONTROLLER_INVOCATIONS.jsonl
tail -n 8 docs/research/h4_pr_multiwidth_resume_20261117/run01/DRIVER_PROGRESS.jsonl
tail -n 8 docs/research/h4_pr_multiwidth_resume_20261117/run01/PROGRESS.jsonl
```

controller/所有workerが残っていない場合の正確な同run再開コマンド：

```bash
bash /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-pr-multiwidth-resume-20261117/docs/research/h4_pr_multiwidth_resume_20261117/START_OR_RESUME.sh
```

wrapperは通常の`run_rpe_schedule_search.py`を`--batch 64 --max-cycles 199`で起動する。残り199 cycleはR06のcycle1から元の200まで。共通queue20、DP3、prefix cacheとm5/m11 checkpointを有効化。flockにより生存中の二重起動はexit75となる。保存済みの重大STOPがあれば同じrunの単純再開では解除しない。

R06の完了Q3は`executor.prior_run_dirs`、READY5構成は新completion state、価格registryとcheckpointはCOPYで引継ぐ。旧runへ書き込まない。過去SOURCEの一部は同じサーバーのGit object/rawでのみ取得できるため、公開remoteだけでの再現とは区別する。

## 結果回収とC向けlocator

- `run01/ROUND_RESULT.json`：方式別の規則採択点・全検証済み最安、各候補、未取得/保留/モデル不適合、配分、MSE上下界、価格とjobへの対応。
- `run01/CONFIRMED_MINIMA.json`：小さな改善や未採択の有効な先読み結果も含む最安。保存開始点を維持し、新規結果と区別する。
- `run01/KAPPA_STATE.json`：要求幅別・L_D別・κ schedule、実幅、選択理由、準備状態と停止理由。
- `run01/PROGRESS.jsonl`、`INVOCATIONS.jsonl`、`jobs/**/DP_PREFIX_EVENTS.jsonl`：実起動数、同時稼働数、DP cache、先読みと採択、実時間。private cacheを公開台帳へ複写しない。
- [SOURCE](SOURCE.json)、[PRELAUNCH_ACCEPTANCE](PRELAUNCH_ACCEPTANCE.json)、[REPORT](REPORT.md)：固定条件、開始点、受入・資源条件。

終了時は元の利用者指示どおり、方式別の開始点、幅0.1での探索後最安、新幅での最安、全幅最安を集計する。条件付き費用、実行時間、直接価格/モデル/外挿、未評価・入力不足・技術UNKNOWN・配分のモデル不適合を区別する。今回は最終集計前にチャットを閉じる場合がある。同じrunから次の担当が回収・公開する。
