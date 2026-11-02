# 将来の明示引継ぎ指示後に使うreceipt手順

今回は実行しない。SOURCE `05a37cebaf18fa28e3a1e8582b86f54e32ba2a66`の限定24 mock検査と軽量reviewは完了し、現在resume01は継続する。

1. 本人の明示引継ぎ指示を受けたら、新しいexclusive evidence directoryとrun02 worktreeをhome配下に用意する。
2. 固定STARTED receiptのPID/start/parent/UIDを照合し、own driverだけへ停止要求を送る。既存cleanupが全worker/observerのwait/reap/FD/pipeを終了させる。
3. 以下のcapture-stopは停止操作を含まず、全6 identityが既に退出したことを2回・2秒以上の間隔で確認する。live parentならSTOP。
4. receiptは全file bytes/streaming SHA・ledger・祖先receipt・mixed SOURCE・cumulative carry・native proofを検証する。科学arrayを読まない。
5. actual停止receiptからSOURCE/profile/input/carry/output/plan/auth/reviewを再結合し、明示launch authorityとfresh gates・独立最終整合review・未使用roots/one-shotを経て一度だけrun02へ移行する。

既存clockは既に固定済みで、再生成やorigin resetを行わない。過去exit code/正確な終了時刻は未記録のまま。
下面は停止後のreceipt操作の例であり、launch commandではない。`h4_receipt_dir`は未使用directoryに設定する。

```bash
h4_checkout=/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-resume02-preparation-20261011
h4_python=/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python
h4_started=/home/AbeHiromu/projects/h4-handoff-evidence/20261011/h4-production-resume01-20261011/STARTED_RUNNING_RECEIPT_resume01.json
h4_clock=/home/AbeHiromu/projects/h4-handoff-evidence/20261011/h4-resume02-preparation-20261011/RESUME01_CLOCK_CONTEXT_v1.json
h4_receipt_dir=/home/AbeHiromu/projects/h4-handoff-evidence/20261011/h4-resume02-preparation-20261011/handoff-after-approval
mkdir -m 700 "$h4_receipt_dir"
env PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 QISKIT_NUM_PROCS=1 QISKIT_PARALLEL=false \
  "$h4_python" -P -B "$h4_checkout/scripts/resource_applicability/run_h4_resume_receipt.py" capture-stop --parent resume01 \
  --started "$h4_started" --clock "$h4_clock" --output "$h4_receipt_dir/RESUME01_NATIVE_STOP_PROOF_v1.json"
env PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 QISKIT_NUM_PROCS=1 QISKIT_PARALLEL=false \
  "$h4_python" -P -B "$h4_checkout/scripts/resource_applicability/run_h4_resume_receipt.py" receipt --parent resume01 \
  --started "$h4_started" --clock "$h4_clock" --proof "$h4_receipt_dir/RESUME01_NATIVE_STOP_PROOF_v1.json" \
  --output "$h4_receipt_dir/RESUME01_CONTINUATION_RECEIPT_v1.json"
```

receipt SHA、carry、import数はこの時点の実測値で決める。今のlive countsを期待値や停止証拠にしない。
17GiB/74805/累積72hを超える場合は起動gateでSTOPし、上限を自動変更しない。
