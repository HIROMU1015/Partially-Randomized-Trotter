# 2026-10-16 M1-S Gate 0 精度停止の原因調査

ユーザーが保存データでの項別調査を指定し、必要な追加計算も許可。診断SOURCE/範囲をd34e2fad5935348282bc43bed8c17073ac471abaで固定。
既存NPZの4 scalar/factor配列だけをdecodeし、exact L1尺度と既存誤差式を一回再集計。state decode、Pauli生成、matrix、W/v/rho、Gate0は0。
H4 linear1.00 Å/STO-3G/8 spin orbitals/rank12、HD G3/G6/G9、deltaなし、保存数値oracle。
全H共通alpha≈3.073e-7 Haとtail lambdaによる正規化が主因。G3/G6は係数law誤差、G9はその線形・二次伝播。
tail尺度だけ、law term countだけの変更は不足。tail＋局所DAG予算を同じ条件付きgammaモデルで準備する余地を示した。
実測counter27781を新しい保証値と認定せず、counterfactualを新契約やscience PASSへ読み替えない。
旧TECHNICAL_STOP/null、17精度条件、SOURCE/input/契約/authorization/markerを保持。再実行せずmandatory STOP。
詳細：[原因・項別予算・候補と境界](../../tracks/algorithm_codesign/m1s_gate0_precision_budget_cause_and_options_20261016.md)。
