# own-run限定の将来command：未実行・最終承認待ち

CPU [3,5,6,7,8,9]、6 worker、mask0x3e8。
以下は別の利用者承認・独立最終review・明示launchとfresh資源/容量検査を満たした後のcommand案。
**現在のreviewはapproved=false**。本commandの掲載によって科学処理を認可しない。
今回はtaskset/worker/runnerを起動していない。reviewの承認文書は別の認可手続きで作成・固定する必要があり、
この未承認ファイルをそのまま使って科学処理を開始できない。

既存science checkoutのsource_rootは変更しない。本bundleのplan/auth/reviewコピーは旧proposalとbyte-identical。
tasksetは新規own processだけを対象とし、既存PID・他jobのaffinity/cgroup/priorityを変更しない。

```bash
/usr/bin/taskset --cpu-list 3,5,6,7,8,9 \
  /usr/bin/env \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  QISKIT_PARALLEL=false QISKIT_NUM_PROCS=1 \
  /home/AbeHiromu/venvs/trotter-common/bin/python -B \
  /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-geometry-resource-observer-fix-20261006/scripts/resource_applicability/run_h4_geometry_input_generation.py \
  --plan /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-input-generation-storage-review-20261006/artifacts/resource_applicability/track_a_h4_input_generation_storage_review/2026-10-06/input_generation_plan_v2.json \
  --authorization /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-input-generation-storage-review-20261006/artifacts/resource_applicability/track_a_h4_input_generation_storage_review/2026-10-06/authorization_proposal_v1.json \
  --review /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-input-generation-storage-review-20261006/artifacts/resource_applicability/track_a_h4_input_generation_storage_review/2026-10-06/stage_review_proposal_v1.json \
  --explicit-launch-input-generation
```

launch直前にCPU/namespace/全祖先memory/pressure/OOM/source bindingと容量をfresh確認する。
nonroot available≥3GiB、free/available inode≥260000、quota非有効または残量十分の確認が必要。
6 workersはeffective memory≥72GiBとsourceのAS/RSS/headroom guardが必要。
不合格/確認不能なら起動しない。容量判定は[最終承認資料](FINAL_APPROVAL_PACKET_v1.md)を参照する。

fixed run ID `track-a-h4-geometry-v2-20261006-run01` とplan内fixed absolute outputを保持する。
6入力生成→freeze→STOPだけで、signal/compileへ続行しない。retry/resumeなし。
72h/output10GiBのcampaign capは不変。10GiB全量の開始前空き確保案は旧履歴に保存し、
今回stage容量は3GiB/260000 inodesで判断する。signal/compile容量は別認可で確認する。
