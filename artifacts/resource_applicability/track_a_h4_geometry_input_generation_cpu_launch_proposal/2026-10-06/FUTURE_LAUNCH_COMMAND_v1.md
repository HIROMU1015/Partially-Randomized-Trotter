# own-run限定の将来command・未実行・未承認

提案CPUは `[3,5,6,7,8,9]`、CPU list `3,5,6,7,8,9`、mask `0x3e8`、requested workers6。
利用者承認未取得。現在のstage reviewはapproved=falseなので、以下の文面で科学処理は認可されない。
このcommandもtasksetも今回は実行していない。

tasksetは新規processのCPU maskだけを指定する。既存PID・他job・共有cgroupを操作しない。
science runnerとそのspawned workersがこの6 CPU以下で動く案で、内部thread/processは各1。
source/planのsource_rootは既存science checkoutのまま。proposal worktreeへ科学sourceを移さない。

```bash
/usr/bin/taskset --cpu-list 3,5,6,7,8,9 \
  /usr/bin/env \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  QISKIT_PARALLEL=false QISKIT_NUM_PROCS=1 \
  /home/AbeHiromu/venvs/trotter-common/bin/python -B \
  /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-geometry-resource-observer-fix-20261006/scripts/resource_applicability/run_h4_geometry_input_generation.py \
  --plan /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-geometry-input-generation-cpu-launch-proposal-20261006/artifacts/resource_applicability/track_a_h4_geometry_input_generation_cpu_launch_proposal/2026-10-06/input_generation_plan_v2.json \
  --authorization /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-geometry-input-generation-cpu-launch-proposal-20261006/artifacts/resource_applicability/track_a_h4_geometry_input_generation_cpu_launch_proposal/2026-10-06/authorization_proposal_v1.json \
  --review /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-geometry-input-generation-cpu-launch-proposal-20261006/artifacts/resource_applicability/track_a_h4_geometry_input_generation_cpu_launch_proposal/2026-10-06/stage_review_proposal_v1.json \
  --explicit-launch-input-generation
```

実行前に必要な条件：

- 利用者がこの具体的CPU集合とown-run起動条件を承認すること。
- 独立最終reviewと承認されたreview文書の更新、利用者の別の明示launchがあること。
- 初期cgroup namespace、全祖先可視性、現在CPU/online/cpuset/topology/loadが条件を満たすこと。
- taskset後のown process CPU集合がallowed_cpus以下で、既存guard/fresh admissionを通ること。
- 6 workerにはfresh effective available72GiB、pressure/OOM等の既存条件が必要。actual wは既存admissionで6以下に決まる。
- 固定outputを置くfilesystemの容量条件を解決すること。今回available約3.717GiBで、全output上限10GiBの空き確保案は未達。

10GiBは実出力量ではなく総上限。これは保守的な運用容量案で、source guardや上限を変更するものではない。
領域削除・別output場所・共有設定変更は今回行わず、quotaとfresh容量は未確認条件としてreviewする。
固定run ID `track-a-h4-geometry-v2-20261006-run01` と固定outputを維持する。
H4の6距離入力を一度generate/freezeしたらSTOP。signal/sampling/build/compileへ継続せず、retry/resumeしない。
