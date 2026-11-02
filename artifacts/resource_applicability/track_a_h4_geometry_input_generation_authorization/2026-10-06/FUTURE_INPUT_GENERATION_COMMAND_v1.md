# 将来の入力生成command・今回は未実行

以下はレビュー対象の文面だけであり、実行していない。
**現在の保存reviewはapproved=false、CPU許可未確定、resource観測条件も未解決なのでlaunchできない。**
最終レビュー、必要なCPU/launch contextとresource条件の解決、更新文書のhash/digest再binding、
利用者の明示launchが別途必要。review承認を自作する手順は含めない。
science source修正が必要と判断された場合、このSOURCE_COMMIT用の草案を実行へ進めず、別source再固定から戻る。

実行対象はレビュー済みactual science checkoutのrunnerで、JSONは認可準備bundleの絶対pathを使う。
CPU集合は最終承認済みのlaunch contextで `process_cpus <= allowed_cpus` を満たす必要がある。
今回affinity/cgroupを変更していない。次の文面もtasksetや共有設定変更を実行するものではない。

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
QISKIT_PARALLEL=false QISKIT_NUM_PROCS=1 \
/home/AbeHiromu/venvs/trotter-common/bin/python -B \
  /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-geometry-parallel-source-20261006/scripts/resource_applicability/run_h4_geometry_input_generation.py \
  --plan /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-geometry-input-generation-authorization-20261006/artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06/input_generation_plan_v1.json \
  --authorization /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-geometry-input-generation-authorization-20261006/artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06/authorization_draft_v1.json \
  --review /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-geometry-input-generation-authorization-20261006/artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06/stage_review_v1.json \
  --explicit-launch-input-generation
```

このcommandは入力生成stageだけを指定する。将来6入力を生成/freezeしたら必ずSTOPする。
signal/sampling/build/compileや次段plan/auth作成へ自動継続しない。
