# 将来の入力生成command・未実行・未承認

以下は文面だけで、今回は実行していない。
保存reviewはapproved=false、allowed_cpus=[]。CPU許可の確定、適合するlaunch context、
必要なauth/review再bindingと独立最終review、利用者の明示launchが別途必要。
新observerのread-only成功はfresh launch admission/CPU許可ではない。
不明namespace/hidden制限、pressure/OOM/資源不足はSTOPし、guardや共有設定を変更して通さない。

actual science checkoutはnew SOURCE_COMMITのsourceとbyte-identicalな専用worktree。
入力生成だけを指定し、JSONはこの新v2 bundleの絶対pathを使う。
以下の文面でtaskset/affinity/cgroupを変更することはなく、承認済みCPU集合へ適合したcontextから将来launchする必要がある。

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
QISKIT_PARALLEL=false QISKIT_NUM_PROCS=1 \
/home/AbeHiromu/venvs/trotter-common/bin/python -B \
  /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-geometry-resource-observer-fix-20261006/scripts/resource_applicability/run_h4_geometry_input_generation.py \
  --plan /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-geometry-resource-observer-fix-20261006/artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06-v2/input_generation_plan_v2.json \
  --authorization /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-geometry-resource-observer-fix-20261006/artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06-v2/authorization_draft_v2.json \
  --review /home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-geometry-resource-observer-fix-20261006/artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06-v2/stage_review_v2.json \
  --explicit-launch-input-generation
```

6距離の入力をgenerate/freezeしたら必ずSTOPする。現草案の認可はsignal/sampling/build/compileを含まない。
signal/compile段の別input-bound plan/認可/review/明示launchなしに継続しない。
