# PR-2 matched-accuracy M1-B1 execution authorization v1

日付：2026-09-30

## status

`M1_B1_EXECUTION_AUTHORIZED_ONCE`

この文書とmachine authorizationは、結果を見ずに固定した194 random＋16 baseline cellのM1-B1を一度だけ
許可する。ただし、authorization bundle自体を別の実行前reviewへ渡すため、このcommitを作っただけでは
launchしない。review承認後も、ここに固定した一回のrun以外へ範囲を広げない。

## 固定identity

- M1-A result commit：`3c1831e326c27c5f679b3820997f27916d26ed9f`
- M1-A result SHA-256：`1f960d7a33296e2dcb74d497e360572b26409dc9aeae01522335e7b91ed81086`
- M1-A result fingerprint：`422f898bba1e3849d0f45830082b76d4f42da436e2b49796e562cd79fc716c9e`
- actual execution source commit：`33f436bb3a7d5b9cefa23604bb22c8d1fb17cd62`
- execution plan v2 SHA-256：`5afc94fac0571b38c74b0b00cfcf68e34e491a5fc65ef579d3ec884d079e5aa5`
- execution plan fingerprint：`17c91d41e77d7c085629b60ea470abc87e9590f448ba9bcdfa4342410cd89607`
- execution authorization JSON SHA-256：`7d188f782354609fec1ea83e289e9872c6a801c3e0e9493f93753fe7cae9d93a`
- execution plan schema v2 SHA-256：`3b503205e32d9a1f72c0fbe9830a93372dbe2c3ff49b225ee255755370c014ff`
- result schema v2 SHA-256：`035d9b1e48d8d02718f8c7218ea49d28297b3b3f29c39c4232cdef83c7c1d081`

source commitに含まれる実行sourceのSHA-256は次である。

| path | SHA-256 |
|---|---|
| `src/trotterlib/pr2_matched_accuracy_m1_b1_contract.py` | `3523a2fc27c3a39e6db46cf70fb985dbc93ee8b9028bddca2bff2a013adb643e` |
| `src/trotterlib/pr2_matched_accuracy_m1_b1_execution.py` | `b7a3a479cbebdd3ff7a93de75c4ba030343769c308401395e208754c3d566987` |
| `scripts/run_pr2_matched_accuracy_m1_b1.py` | `07f34e4f1a22e93ace21d26ce122b7931311f8d94dfebd2d79f1086d9fdc9371` |
| `tests/test_pr2_matched_accuracy_m1_b1_execution.py` | `971caf4c9b5025bfae6779ff15a53656fecebc068152192a72b1cea8555dc712` |
| `artifacts/pr2_matched_accuracy_m1_b1_contract/2026-09-30/pr2_matched_accuracy_m1_b1_result_schema_v2.json` | `035d9b1e48d8d02718f8c7218ea49d28297b3b3f29c39c4232cdef83c7c1d081` |

runnerはauthorizationのsource hash集合がplanのsource hash集合と完全一致すること、source commitが実行時HEADの
ancestorであること、各source byte、plan SHA/fingerprint、result schema SHA、permission、resource capが一致する
ことを科学計算前に検査する。

## 認可する計算

- M1-Aでaccuracy適格だったB2 145＋B3 49、計194 fingerprintだけを使う。
- signal、bias、normalization、shotはM1-Aを再利用し、再評価しない。
- random各cellは32 trajectoryを一度だけsampleし、同じtrajectory列をcosine/sineで共有する。
- random wrapperは`194 × 32 × 2 = 12,416`。
- B0 12＋B1 4は全16 cellを二軸でcompileし、32 wrapperとする。
- 総上限は12,448 full wrappers、最大6 spawned process worker、各BLAS thread 1とする。
- B0のaccuracy不適格4 cellはcompile completenessにだけ含め、matched-accuracy frontierから除く。
- development H4 linear 1.00 Å、STO-3G、DF rank 12の既存snapshotだけを一回loadする。

candidateごとのrequest seedと、benchmark実装が生成する32 trajectory seedはexecution plan v2で固定済みである。
cacheはsource commit/candidate単位、wrapper result identityはcompiler/candidate/axis/trajectory index/seedまで含む。
checkpointはcell全体のtask fingerprintが一致するときだけ再利用する。

## 禁止事項

- candidateの追加、削除、置換、signal再評価。
- 33件目以降または追加96 trajectory。
- held-out H4 1.30 Åのpath resolve、stat、hash、load、signal、cost、ranking。
- transfer、winner精密化、S3、新分子、新PF、新時刻、threshold変更。
- GPU query、allocation、kernel。
- science runnerによる研究四分岐の自動選択。

## terminal statusと停止

result schema v2が許すterminal statusは次の2件だけである。

- `M1_B1_COMPILE_MAP_COMPLETE_AWAITING_REVIEW`
- `IMPLEMENTATION_GATE_FAILED`

成功時も`research_decision=null`、`automatic_next_stage=null`を維持する。32-trajectory actual compiled resource
mapを作った時点で停止し、別reviewで初めて次のいずれかを判断する。

- `CONTINUE_RESOURCE_STUDY`
- `NARROW_TO_TECHNICAL_NOTE`
- `STOP_DUPLICATIVE`
- `COMPILE_RESULT_INCONCLUSIVE`

## 実行command

実行前reviewが承認した場合だけ、source commitの子孫でsource byteが不変なclean worktreeから次を実行する。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
'/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python' \
  scripts/run_pr2_matched_accuracy_m1_b1.py run \
  --project-root . \
  --authorization artifacts/pr2_matched_accuracy_m1_b1_contract/2026-09-30/pr2_matched_accuracy_m1_b1_execution_authorization_v1.json \
  --plan artifacts/pr2_matched_accuracy_m1_b1_contract/2026-09-30/pr2_matched_accuracy_m1_b1_execution_plan_v2.json \
  --output-dir artifacts/pr2_matched_accuracy_m1_b1_execution/2026-09-30 \
  --workers 6
```

中断時だけ同じcommandへ`--resume`を加えられる。別output、別worker数、別plan、別source、別authorizationへの
切替は再開として認めない。
