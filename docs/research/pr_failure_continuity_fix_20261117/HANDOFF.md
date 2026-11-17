# 修正SOURCEと引継ぎ

コードSOURCE：`314524a913ba6e66b9069ab1514b1df765f9a5e2`。
branch：`codex/b2-rpe-pr-failure-continuity-20261117`。
基点：R08入力 `8de068f60e29f0936fdfb13ada9026e709ce3e41` / code `07b9c1ce826ba67e7e8237b1d1c781a2e1357e6d`。
worktree：`/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-pr-failure-continuity-20261117`。

## 変更の利用

R08と同じSOURCE系列へのコード差分はこの3commitの範囲。旧runの資料やrawを変更するcommitではない。

```bash
git cherry-pick 8de068f60e29f0936fdfb13ada9026e709ce3e41..314524a913ba6e66b9069ab1514b1df765f9a5e2
```

稼働中のcheckoutへ適用しない。他担当の分散統合版をmergeしたものでもなく、そのbranchを変更していない。
コードを採用する側で新SOURCEを固定し、元の通常入口を使う。新queue/callerや追加の承認証明は不要。

```bash
PYTHONUNBUFFERED=1 /path/to/controller-python -B scripts/run_rpe_schedule_search.py \
  --problem FIXED_PROBLEM.json --round FIXED_ROUND.json --run-dir NEW_RUN_DIRECTORY \
  --batch 64 --max-cycles REMAINING_CYCLES > NEW_CONTROLLER.log 2>&1
```

上は入口の書式であり、実行を起動するための新しい科学planではない。R08のSOURCEやroundのcycle/予算をそのまま上書きしない。

## 失敗時の観測

`RUNTIME_ERRORS.jsonl` を最初に読む。error_type/message/traceback、method/phase、時計、PID/threadを保存する。
`controller_state=DRAINING_AFTER_EXCEPTION` は正常探索ではなく、所有する走行中のstage/workerを回収している状態。
同じ失敗がmethod・phase・driverの複数文脈で現れる場合がある。最初の例外を保持し、二次例外は別recordとして残す。

`ROUND_STAGES.jsonl` のERRORにはerror locator、`DRIVER_PROGRESS.jsonl` にはBATCH_EXCEPTION/DRIVER_ERROR、
`KAPPA_STATE.json.stop` にはRUNTIME_EXCEPTIONが残る。state保存はproducer回収後に行うため、回収中の判断には即時error logを使う。

`STAGE_YIELDED_AFTER_ORCHESTRATOR_FAILURE` / `PREPARATION_YIELD_AFTER_FATAL` は、未開始stage/次候補を止めた記録。
元のSTARTED/RETURNは維持する。成功済みRETURNを再利用し、失敗RETURNや曖昧なSTARTEDを科学retryにしない。

## UNKNOWNの境界

原因未分類・数値・identity・共通SOURCE・保存障害はglobal保護を維持。
元REQUEST/routeを確認したAPI前scope拒否（科学call 0）と、明示SIGINT/SIGTERMによるOUTCOMEなしの終了だけを局所保留する。
SIGKILL/OOM疑い・SIGSEGV・MemoryErrorを自動的に局所扱いしない。UNKNOWNはMSE適合点にしない。
局所保留は既存 `execution_holds` へ残り、同じ候補を再起動しない。既存適合点・他候補は使える。

## 実R08の引継ぎ前に必要なこと

R08：host `GPUA100-gleap02`、tmux `h4-pr-width-r08`、controller PIDは起動時716100。
run：`/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-pr-candidate-isolation-20261117/docs/research/h4_pr_candidate_continuity_20261117/run01`。

まずそのcontroller/所有worker、直近stageとreceiptを確認する。同runのcontrollerを二重起動しない。
今回の担当は停止・signal送信・再起動・SOURCE途中変更を行っていない。

1. 旧processの終了/所有権と、走行中stageの保存状態を確認する。
2. 保存信号・価格・holdout・Q3をSOURCE/input/bindingで照合する。同等の完了分を取り直さない。
3. 旧SOURCE・raw・STOPを保持し、新SOURCEと別runを固定する。SOURCE不整合をstateの書換えで消さない。
4. 過去の失敗stage・不明なSTARTED・科学STOPは保留する。今回の修正を科学retry許可と扱わない。
5. descriptor補完後requestの過去recordは、完全一致が一意な元keyだけ再利用する。旧recordを新keyへ付け替えない。
6. 旧R08の16:00 ERRORの具体的例外は未確定。自然終了時のtraceback等を回収して確認し、修正で解消済みと推測しない。

今回、新規H4信号・compile・fit・実Q3・科学retryは0件。新科学runは起動していない。
元の広域探索の承認範囲・数値条件・残りcycle/budgetは拡張していない。

## 検証の再現

```bash
/path/to/controller-python -B -m unittest discover -s tests -p 'test_rpe_failure_continuity.py' -v
/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python \
  -B -m unittest discover -s tests -p 'test_rpe_failure_continuity.py' -v
```

追加10件、既存82件の合計92 PASS。nativeで追加10 PASS。
通常CLIの再現証拠は [checks/NORMAL_ENTRY](checks/NORMAL_ENTRY/)、結果は [REPORT](REPORT.md) と3本のREGRESSION log。
公開証拠の一部は一時directoryのpathを記録しているが、入力は全て人工。旧H4科学結果の新規再解析ではない。

ロードマップへの差分は、失敗の可視化・局所保留・協調回収の技術検証完了。
R08の具体的例外解消、実runへの適用、最安更新や科学的資格の追加は未認定。ロードマップ本体は統合担当へ渡す。
