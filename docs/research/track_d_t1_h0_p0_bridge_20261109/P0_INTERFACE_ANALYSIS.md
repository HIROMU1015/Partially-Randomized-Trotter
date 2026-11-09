# P0境界の保存証拠と最小ruleの対応

2026-11-09 JST。[全結果・条件](REPORT.md)。Aの対象はC7-2固定54人工case、108 raw/compiled record。compiled54列は全てnative取得可能、raw16列が適格、raw38列はordinary制御・Gaussian・CSWAP等の未展開gateを含み対象外だった。全54のcompiled比較を行い、最初の3列だけで終了していない。

## 1. Native列の取得・照合

入力は保存 `operation_traces.json`、`case_metrics.csv`、`summary.json`、SOURCE。hashを709d6ed9のGit bytesと照合した。opsは[name,qubits,parameters]でclassical indexを持たないため、既存[hadamard_wrapper](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c583abba520a711895d21b7ec9590688349a7bd1/src/trotterlib/track_d_minimal_reference.py#L50)のancilla→c0と保存weighted-wire-finishを使う固定再構成に限定した。ncが0/1以外、measurement対象がancilla以外なら拒否する。空compiled列の幅もCSVから明示し、列から推測しない。

70適格native record全てでRZ/CX count、RZ/CX/total depth、size、幅が既存値と一致した。未対応rawをnative gate一個として数えず、未展開列をcompiled価格予測の成功例へ混ぜなかった。[全54結果](../../../artifacts/track_d_t1_h0_p0_bridge/2026-11-09/run01/artificial_trace_comparison.json)。

## 2. B1〜B5の分類

| 接口 | 保存証拠と今回の確認 | bridgeでの扱い |
|---|---|---|
| B1 同方向CX対 | cx_adjacentはraw2→P0 0。両wireで隣接する同一control/target対を取り除けば一致 | 実装。逆方向CX、wireに介在するgateは消さない |
| B2 control側RZ | cx_commuting_control_rzはCX2のまま。数学的な可換移動で0へする最適化はこのP0が行っていない | RZをblockerとして残す。数学的可能性と実P0を区別 |
| B2 target側RZ | cx_noncommuting_target_rzもCX2を保持 | 同じくblocker。axisやparameterを捨てない |
| B3 parity | center_repeat_zz_controlledはraw8CX→bridge6CX＝P0、全7資源も一致。physicalは4→2CXでCXは一致するがRZ等は不一致 | 展開済みnative parity端のCXだけにruleを適用。RZZ loweringはしない |
| 1Q/signed/zero | physicalの同support連続やopposite signedではP0が1Q runを再合成／除去する。RZ(0)介在もP0では消えるが今回ruleでは残る | 未実装。特別な閾値・2π式を追加せず残差を保存 |
| B4 Gaussian／actual inverse | actual_inverse_pairのcompiledはsize0でT0照合PASS。ただしrawのXXPlusYY等は未展開で、今回before-priceの成功例にはならない | 既存Strongのrelative transition、同basis共有、actual inverseを保全。新合成器なし |
| B5 Reference shell | isolated CSWAPはP0 8CX、既存scalar-only四CSWAP shellは28CXで単体和32と異なる。whole-pathのnative compiledも全件照合 | CSWAP単体価格の加算をしない。lane端／χ／readoutとのP0 bridge未閉包 |

Qiskit1.3の[InverseCancellation Rust SOURCE](https://github.com/Qiskit/qiskit/blob/1.3.0/crates/accelerate/src/inverse_cancellation.rs)は、DAG上でcollectしたrunをqargs等で分け、同じself-inverse gateを偶奇で除去する。opt1の[反復pass](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/transpiler/preset_passmanagers/builtin_plugins.py#L535)は1Q再合成／inverse cancellation等をdepth・size固定点まで繰り返す。今回はこのうち同方向CXの両wire隣接に対応する部分だけを実装した。Rust／P0全体の複製ではない。

## 3. Countとdepthの残差

| 資源 | 適格raw16列でbridgeが保存P0と一致 |
|---|---:|
| CX count | 15/16 |
| CX depth | 15/16 |
| RZ count | 10/16 |
| RZ depth | 10/16 |
| Total depth | 10/16 |
| Circuit size | 10/16 |
| Qubits | 16/16 |

全7指標一致は10/16。この比率は固定人工case集合の分類で、全PR pathの精度率／coverage／成功確率ではない。

不一致6列は `center_changed_support_physical`, `center_opposite_signed_physical`, `center_repeat_plus_physical`, `center_repeat_zz_physical`, `diagonal_explicit_zero_directional`, `diagonal_explicit_zero_uncontrolled`。最後のzero-directionalだけはCXも2本残りP0との差が出た。CXだけの一致が多くても、RZ／size／depthまで全価格が一致したことにはならない。全compiled54列には今回ruleでさらに消えるCXはなかった。

## 4. Chunkの情報損失と指定3 P0

全native列 `CX01,CX12 | CX12,CX01` を固定し、full／first_chunk／last_chunkを各一回compileした。

| 対象 | 入力CX | bridge CX | P0 CX | P0 total depth | correctness max-entry |
|---|---:|---:|---:|---:|---:|
| Full | 4 | 0 | 0 | 0 | 0 |
| First chunk | 2 | 2 | 2 | 2 | 0 |
| Last chunk | 2 | 2 | 2 | 2 | 0 |

二chunkを独立最適化した価格和はCX4／depth4で、full P0と一致しない。bridgeの状態を保持して二chunkを順次入力すると、中央の対が消え、その後初端／末端の対も消える。逆に、独立chunkをT0のmatrix/countだけへ確定するとgate opcode／順序を失い、この後続相殺は復元できない。

この3 callが必要だったのは、既存54caseに同じ固定native wordと両chunkのcompile出力を一緒に比較する記録がなかったためである。追加caseは生成せず、SOURCE・raw/compiled hash・小規模位相付きcorrectnessを[保存](../../../artifacts/track_d_t1_h0_p0_bridge/2026-11-09/run01/artificial_new_P0.json)した。H4 compileは0。

## 5. 保証の階層

T0は与えたnative wordを厳密に数える。bridgeは定義したCX reductionの結果を同じT0で厳密に数える。保存caseや指定3 P0での一致は有限のP0対応証拠である。純CX列では他の翻訳・1Q処理が関与しない条件下で構造的対応を考えられるが、今回全native列／full wrapperの閉包証明は採択しない。

Gaussian・1Q run・zero/special angle・CSWAP／χの未確認処理は推測せず、残差を0にしない。旧C7-2のshell＋max(lane)反例（例：q2_sameで全体162、shell42＋max(lane115)=157）も保持する。差+5/+3をH4へ共通定数移送しない。
