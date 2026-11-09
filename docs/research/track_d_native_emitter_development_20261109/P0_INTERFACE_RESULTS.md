# 1Q・Gaussian・parity・CSWAP接口の対応結果

[全結果](REPORT.md)／[全54機械結果](../../../artifacts/track_d_native_emitter_development/2026-11-09/run01/artificial_results.json)。旧人工SOURCEは709d6ed9c234bf6d5a8848d3a30be69d88fd6476の公開索引に固定した。operation traceに欠ける非native gate角を推測せず、hashに一致する保存raw QPYを使った。compiled QPYの再生成や新P0はない。

## 1Q・zeroの原因と最小処理

旧CX-only bridgeはRZ(0)もblockerとして残し、signed連続RZ・SX/Xを再合成しなかった。新[1Q bridge](../../../src/trottertracks/track_d_p0_1q_bridge.py)はwireごとのnative 1Q runを2×2行列へ順序通り積み、Qiskit1.3の `Optimize1qGatesDecomposition._resynthesize_run` だけを呼ぶ。固定basisで候補はZSXX、target／error_mapはなし。返されたrunが厳密に短い場合だけ採用し、declared global phaseを加える。一般1Q optimizer、独自special-angle閾値、全DAG passは実装しない。

その後、旧CX bridgeの両wire隣接・同方向CX相殺を使い、sizeが減る間だけ反復する。非zeroのcontrol-RZを数学的に可換移動してCXを消す規則は追加しない。zero除去後に露出したCXや、SXを含む再合成と後続相殺は処理できる。

Qiskit公式[1Q pass SOURCE](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/transpiler/passes/optimization/optimize_1q_decomposition.py)のbasis選択・2×2合成・置換条件と、[opt1 pass構成](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/transpiler/preset_passmanagers/builtin_plugins.py)を確認した。今回のstrict-shorter条件はtarget-free native domainの限定利用であり、P0の全pass順序・translation・同長書換えを再現するものではない。

| 旧不一致case | 新bridge対保存P0：全7指標 |
|---|---|
| center_changed_support_physical | 全差0 |
| center_opposite_signed_physical | 全差0 |
| center_repeat_plus_physical | 全差0 |
| center_repeat_zz_physical | 全差0 |
| diagonal_explicit_zero_directional | 全差0 |
| diagonal_explicit_zero_uncontrolled | 全差0 |

RZ(2π)のgate除去もglobal phaseを保持して人工作用比較する。微小な合成位相を0にclampしない。部分最適化は公式合成器の浮動小数点規約を使うため、全角での数学的な完全同値証明とは区別する。

## Gaussian／parity／CSWAP

| 接口 | 今回の到達点 | 残る条件 |
|---|---|---|
| Gaussian・relative transition | 元StrongBuilderのbasis／relative／same-basis規則を保持。標準XXPlusYY等をloweringし、人工headと保存Gaussian／A† caseの作用・価格が一致 | 任意H4角・frame数・translation tie-break・長い逆順列でのP0閉包は未検証。新Gaussian optimizerなし |
| Parity・controlled-RZZ | 標準definitionからnativeへ。directional／ordinary／physicalをそれぞれ保持し、native CX端と1Q runだけを処理 | 異なるsupport／可換変形の一般最適化を足さない。保存case外は未資格 |
| Actual inverse | 小QC逆順＋各gate.inverse、負のdeclared phase。保存actual_inverse_pairはnative CX56から最終size0、保存P0と一致 | このcaseは30反復を要した。低い固定反復数やbounded frontierの根拠にはならない |
| CSWAP shell・χ・readout | 単体CSWAPは8CX、四CSWAPのscalar shellは28CX。新bridgeは保存shell／whole wrapperの全資源に一致 | 単体和32を使わない。共通ancilla／A† B lane／χ／readoutを全wire依存として扱う |

全54caseでRZ/CX count・三depth・size・幅が同時一致した。これは有限人工集合での強い技術的改善であり、未知の残差が0という証明ではない。旧q2_sameのdepth162に対しshell42＋max(lane115)=157となる反例や、別caseの+3残差を保全する。H4に定数補正として移送しない。

## 資格の区別

**数学的・計数上の規則**：native dependency列の整数count／depth、合法なwire mapping、CX²=I、actual adjointの順序、固定native templateのmax-plus連結。1Qの理想的unitary変換は合成phaseを含めて扱う。

**保存P0で有限確認**：新lowering＋strict-shorter 1Q＋CX反復による固定54caseの全7資源一致、旧6不一致の解消。raw→native／bridgeの位相付き作用比較はcomponent atol1e−10／rtol0、phase fitなし。

**条件付き／未資格**：全P0 schedule、未知gate／symbolic／conditional、calibration／target／routing／別Qiskit版、任意Gaussianや任意角、H4長qの絶対P0価格。`P0_UNQUALIFIED`を維持する。nativeのRZをT countへ、topology-free depthを実時間へ換算しない。A/F/Sの係数、既存P0価格、208−224qの位置づけは不変である。
