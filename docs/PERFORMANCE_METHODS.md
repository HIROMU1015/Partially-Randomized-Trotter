# 並列化・高速化の再利用索引

## PR-Pauli LD2 m9：固定coreのworker内再利用

[scripts/run_pauli_ld2_m9.py](../scripts/run_pauli_ld2_m9.py) の `accelerated_observed`：
完全event JSON・label map・nをkeyに既存event_matrixをmemoizeし、既存fast canonical writerを
coreのfingerprint context全体（wrapper内部を含む）へ適用。独立pathは別process、段階log/親observerで
終了codeを保存。旧科学SOURCE・sampler・D2・compiler・gateは無変更。小型dense matvecはBLAS1 thread/process。
[準備/検証](research/d_pauli_ld2_m9_longpoint_results_20261115/SOURCE_PREPARATION.md)、
[4 test](research/d_pauli_ld2_m9_longpoint_results_20261115/TESTS.txt)。toy同等性とreceiptまで確認。
実m9の独立2 path取得wallは3825.209秒、worker peak RSSは約7.5〜7.6百万KiB。speedup比は未測定。Python3.11.0rc1の非同期定期frame dumpは使わない。

m9の保存raw公開照合は [publish_pauli_ld2_m9.py](../scripts/publish_pauli_ld2_m9.py) で
2 pathを2 processに分け、元receipt検査後のprices/scalarだけを親へ戻す。
実行中SOURCEを変更せず、raw SHA・axis/SOURCE/seed/metrics・exact集約を保持する。
[適用条件と残る逐次処理](research/d_pauli_ld2_m9_longpoint_results_20261115/PUBLICATION_ACCELERATION.md)、
[人工2 test](research/d_pauli_ld2_m9_longpoint_results_20261115/PUBLICATION_TESTS.txt)。

実m9のraw公開照合は2 PID・各約4.7秒、全体7.310秒で完了。科学SOURCE650dを保持し、sampling/compile/fit追加0。
[実測と残る逐次処理](research/d_pauli_ld2_m9_longpoint_results_20261115/REPORT.md)。
