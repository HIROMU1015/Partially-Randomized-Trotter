# 並列化・高速化の再利用索引

## PR-Pauli LD2 m9：固定coreのworker内再利用

[scripts/run_pauli_ld2_m9.py](../scripts/run_pauli_ld2_m9.py) の `accelerated_observed`：
完全event JSON・label map・nをkeyに既存event_matrixをmemoizeし、既存fast canonical writerを
coreのfingerprint context全体（wrapper内部を含む）へ適用。独立pathは別process、段階log/親observerで
終了codeを保存。旧科学SOURCE・sampler・D2・compiler・gateは無変更。小型dense matvecはBLAS1 thread/process。
[準備/検証](research/d_pauli_ld2_m9_longpoint_results_20261115/SOURCE_PREPARATION.md)、
[4 test](research/d_pauli_ld2_m9_longpoint_results_20261115/TESTS.txt)。toy同等性とreceiptまで確認。
実m9の所要時間・speedupは取得前未測定。Python3.11.0rc1の非同期定期frame dumpは使わない。
