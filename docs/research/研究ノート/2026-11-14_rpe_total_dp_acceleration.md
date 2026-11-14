# Total DP高速化

2026-11-14。実行中固定2件を保全する独立worktreeで、整数質量のoutcome-wise集約、process並列化、残余・momentの遅延約分を実装した。旧SOURCEのgeometry/pruning/decoder/finite scopeを再利用し、全output一致を人工データで確認。43件PASS、人工219万遷移で80.9秒→3.8秒。

[REPORT](../rpe_total_dp_acceleration_20261114/REPORT.md)。実H4へ未適用、新科学計算なし。現jobは旧SOURCEで継続。高速版への切替・再実行は固定SOURCE/各1回条件に関わるため、具体的な指示を得てから行う。

ユーザーが切替と代替2件を明示許可。旧runは0a6f8e4a5に保全停止し、新SOURCE67950b267で同じ科学入力を各1回実行した。約235秒で2件完了、exact mass/loss/moment gateと旧coarse/lemma4件照合PASS。[結果](../pauli_total_q3_accelerated_results_20261114/REPORT.md)。十分N/G等はnull、公開後mandatory STOP。
