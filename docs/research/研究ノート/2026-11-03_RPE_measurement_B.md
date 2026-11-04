# 2026-11-03：RPE固定request B

前段で候補A–Dを比較し、先に実回路の条件付き平均を切り分けるBを提案した。
ユーザーの次作業への進行許可を受け、SOURCEを4d7c455db3ed7d3e4bc9e2e1fd4dbeba9f46b272で固定し1回実行した。

linear H4/1Å/STO-3G/legacy DF12、8qubit・4電子sector、physical ground state、
L_D3/12、δ=.02、q1/q8。Partial r1/K2のorder0・先頭order2固定profile、Full r0/K0。
6request・12axisのideal meanは1e-10内で一致（最大8.66e-15）、
control branch作用の最大残差1.84e-14。再構成physical stateの保存fingerprintも一致した。
新state solve1、sampling/quantum shots/compile/価格再解析0。

意味は少数の同一state/requestでの符号・scalar・control/Hadamard接続の局所支持である。
分布平均・fresh-IID・18round成功率、原Full fingerprint欠測、V1の半径問題、R1費用未確定は残る。
次はAを別承認の候補にする。今回の結果からsamplingやレビューを自動開始せずmandatory STOP。

[結果・条件・限界・rawへの入口](../rpe_fixed_request_measurement_b_20261103/REPORT.md)。
