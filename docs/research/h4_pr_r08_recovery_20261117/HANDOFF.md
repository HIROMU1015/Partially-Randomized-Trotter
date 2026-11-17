# R08修正版への引継ぎ

新コードSOURCE：`fdb4453a0fae36c8b8734a225a19316446dee62e`。失敗継続修正SOURCE314524a91に、R08限定の所有権待ち・metadata移行callerを加えた。科学数値kernelは変更していない。

- host：`GPUA100-gleap02`
- branch：`codex/b2-rpe-pr-failure-continuity-20261117`
- worktree：`/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-pr-failure-continuity-20261117`
- round ID：`H4_PR_BOTH_MULTIWIDTH_CONTINUITY_R08_20261117`（R08を継続）
- 新run：`docs/research/h4_pr_r08_recovery_20261117/run02`
- tmux：`h4-pr-r08-recovery`（20:40 JST起動、待機PID1383390、記録 `SERVER_START.json`）
- 固定plan：`HANDOVER_PLAN.json`。元SOURCE07b9c1ce8、入力hash、元cycle上限200、queue20/DP3を記録。
- 実入口：`START_OR_RESUME.sh` → `scripts/resume_pr_r08_after_drain.py` → **通常** `scripts/run_rpe_schedule_search.py`。

## 最初の状態確認

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-pr-failure-continuity-20261117
tmux list-panes -t h4-pr-r08-recovery -F '#{pane_pid} #{pane_dead} #{pane_current_command}'
cat docs/research/h4_pr_r08_recovery_20261117/HANDOVER_STATUS.json
tail -n 20 docs/research/h4_pr_r08_recovery_20261117/HANDOVER.log
```

`WAITING_FOR_OLD_OWNER` なら、旧 `h4-pr-width-r08` の科学stageの自然終了待ち。新科学controllerは未起動。旧runは以下。

`/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-pr-candidate-isolation-20261117/docs/research/h4_pr_candidate_continuity_20261117/run01`

`FIXED_SOURCE_RUNNING` になったら、新runの `CONTROLLER_PID.json`、`CONTROLLER.log`、`DRIVER_PROGRESS.jsonl`、`PROGRESS.jsonl`、`completion/PROGRESS.jsonl` を確認する。PID生存だけで進捗・gate PASSとしない。SOURCE/input commitはrun02/SOURCEとstatus、即時例外はrun02/RUNTIME_ERRORSに残る。入力は所有権解放後の最終metadataを使い、科学再開前にcommit/pushされる。

## 起動／同じrunの再開

**上の状態確認を先に行う。稼働中は起動しない。** 同一入口はHANDOVER.lockで二重起動をexit75にし、旧CONTROLLER.lockも新controller終了まで保持する。旧runのcontrollerを別途起動しない。

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-pr-failure-continuity-20261117
tmux new-session -d -s h4-pr-r08-recovery 'bash /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-pr-failure-continuity-20261117/docs/research/h4_pr_r08_recovery_20261117/START_OR_RESUME.sh'
```

既存run02があればSOURCEと固定入力を検査し、元stateのcycleから残りを計算する。元の通常CLI・同じrun directory・同じprior run一覧を使う。既存のRUNTIME_EXCEPTIONや科学STOPを自動解除しない。HANDOVER_STOP/UNKNOWNの場合は記録を確認し、同じ失敗科学stageをretryしない。wrapperは1回だけ起動し、失敗時の無限再起動はない。

## 受渡し・保全

316保存Q3は元Executorでkey/route/bindingを再検査して再利用。旧準備recordと失敗理由、scalar、lowering/compiler、full wrapper、価格の直接/MODEL区分を変更しない。価格registryとstep registryは同じlocatorで引き継ぎ、prefix cacheも同じdirectoryを使用。別幅の信号・価格を流用しない。

実metadataではDF LD4/.05・LD8/.1・LD8/.05のlambda_source不一致を元guardが拒否する。これらはcandidate holdとして残す。進行中LD4/.1の旧stageは終了まで観測する。旧ERRORの具体的tracebackは `OLD_CONTROLLER_EXCEPTION.txt` に回収されるまで未確定。

REPORTと `PRELAUNCH_ACCEPTANCE.json`（移行時生成）が技術的受入根拠。新runのROUND_RESULT/CONFIRMED_MINIMAができた時点で、実際に評価した集合・条件付き価格・未評価/保留・採択/先読みを区別して次チャットまたは既存後処理が公開する。現時点で探索完了・実H4速度倍率・新最安を認定しない。
