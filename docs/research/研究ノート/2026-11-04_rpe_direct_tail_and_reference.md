# 2026-11-04 RPE A/C：直接tailの95%条件と保存DF referenceを確認

公開SOURCEd29a86963f91044a95217d7d2e2ee9703d2f1dc1への別途実行許可後、A+Cを各1 pass実行した。
Linear H4/1Å/STO-3G/保存legacy DF12、Partial L_D3/δ=.02、保存state ray・sector70、
全18round・保存q/r/K/N/α/β/ε/E_refを維持。

A160-bitは旧36軸U_cat/N/εから直接Hoeffding上界を評価し、和≤0.0496929699367478180、
成功下界≥0.950307030063252182を得た。保存αや旧Σα>.05の結果は変更せず、別の直接上界によって95%条件が成立した。
C256-bitは保存係数のexact Hermitian性と完全sectorを確認し、固定lower shiftで70正pivot、Rayleigh上界を得た。
η_ref≤1.00000000000000001e−8 Ha、数学的decoderの保存DF sector ground energy誤差上端≤0.000138689200360119854 Haで保存目標内。
最小性は正定値下界で識別し、小residualだけでgroundと判断しない。

BのFloat64一様誤差、fixed PRNG/IID、実回路・compiled/noise・実state preparation、元分子energy等の範囲外は未保証。
global false/UNDETERMINED、両bridge null、旧SOURCE/raw/STOPを保全する判断とした。
[実行報告・原結果](../rpe_three_gaps_execution_20261104/REPORT.md)。
retry/精度追加/shift変更・sampling/compile/cost/eigsh/eigh0。A/C追加実行は不要。
次はユーザーがBの静的証明準備の範囲を指定する。公開照合後mandatory STOP。
