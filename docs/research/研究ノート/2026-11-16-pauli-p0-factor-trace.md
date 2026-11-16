# 2026-11-16 H4 Pauli P0の係数追跡

ユーザー承認により、49a6d7adeのNEXT_CHECKSを科学SOURCE655e74212で実行した。
保存operator3件を再解析し、元入力を1回復元、LD9→8→7の順で実factor trace各1件。
元P0/gate1e-9/global EQ_TOLERANCEを変更せず、全default JWが保存結果と完全一致した。

gate超過の主因はLD7でJW部分和除去、LD8/9でweighted Fermion加算とJW部分和除去。
途中除去なしの局所診断は保存tailと約1e-19以下で一致。最初の差はone_bodyにあるが、
その微小寄与と主因を区別した。新しい本番辞書の採用、signal・価格較正・Q3は行わない。

[REPORT](../pauli_p0_factor_trace_results_20261116/REPORT.md)／
[最小修正案](../pauli_p0_factor_trace_results_20261116/MINIMAL_FIX.md)／
[B2 HANDOFF](../pauli_p0_factor_trace_results_20261116/HANDOFF.md)。
元の原因未確定の記録・STOPは歴史的証拠として保持する。次は別途承認された新SOURCEでの局所修正/P0再資格。
